import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mirror import decode as D, trends as T, trend_insights as I, trend_discovery as Q, food_scope as FS


class InsightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = patch.object(D, 'DB', Path(self.tmp.name)/'decode.db')
        self.db.start(); T.init(); I.init()
        self.now = time.time()

    def tearDown(self):
        self.db.stop(); self.tmp.cleanup()

    def row(self, i=0, age=1, creator=None, likes=10, name='말차'):
        row = D.normalize(dict(id=str(i),caption=f'{name} 모찌 쫀득한 식감 시음',creator=creator or f'creator{i}',likes=likes,views=None,
                               published=self.now-age*86400), 'tiktok')
        row['food_category']='음식'; row['food_p']=.95
        row.update(history=[],reaction_history=[],share_history=[],rate=None,reaction_rate=None,share_rate=None)
        tag=dict(entities=[dict(name=name,evidence=name)],genre='떡·모찌',themes=[dict(name='식감·단면',evidence='쫀득한 식감')],summary='말차 모찌 식감 시음')
        with D.db() as c:
            c.execute('insert or replace into trend_tags values(?,?,?,?,?)',(row['id'],D.digest(D.state(row)),I.VERSION,json.dumps(tag),self.now))
        return row

    def test_ai_labels_require_source_quotes_and_valid_ids(self):
        row=self.row();item=dict(id=row['id'],entities=[dict(name='한정선',evidence='한정선 인기'),dict(name='말차',evidence='말차 모찌')],genre='떡·모찌',themes=[dict(name='오픈런·희소성',evidence='대기 3시간')],summary='말차 모찌')
        result=I.parse(json.dumps(dict(items=[item])),[row])[row['id']]
        self.assertEqual(result['entities'],[dict(name='말차',evidence='말차 모찌')]);self.assertEqual(result['themes'],[])
        for bad in [item|{'id':'wrong'},item|{'genre':'없는 장르'},item|{'entities':[5]}]:
            with self.assertRaises(ValueError):I.parse(json.dumps(dict(items=[bad])),[row])
        with self.assertRaises(ValueError):I.parse('[]',[row])

    def test_old_high_views_and_new_discovery_do_not_become_current(self):
        rows=[self.row(i,age=180,likes=100000) for i in range(6)]
        for row in rows:row['first_seen']=D.now();row['observed']=D.now()
        food=next(g for g in I.snapshot(rows,self.now)['groups'] if g['kind']=='food')
        self.assertEqual(food['current'],0);self.assertEqual(food['growing'],0);self.assertEqual(food['current_ids'],[])
        self.assertEqual(food['status'],'과거·시점 미확인')

    def test_future_and_unknown_dates_do_not_count_as_recent(self):
        rows=[self.row(1,age=-1),self.row(2)];rows[1]['published']=None
        self.assertTrue(all(g['current']==0 for g in I.snapshot(rows,self.now)['groups']))

    def test_popularity_requires_peers_and_distinct_authors(self):
        rows=[self.row(i,creator='same',likes=100+i) for i in range(6)]
        food=next(g for g in I.snapshot(rows,self.now)['groups'] if g['kind']=='food')
        self.assertEqual(food['creators'],1);self.assertEqual(food['status'],'표본 부족')
        small=I.snapshot(rows[:2],self.now)['groups'][0];self.assertIsNone(small['score'])
        rows=[self.row(i,likes=None) for i in range(6)]
        food=I.snapshot(rows,self.now)['groups'][0];self.assertIsNone(food['score']);self.assertEqual(food['status'],'최근 관측')

    def test_zero_is_a_measured_zero_not_an_unknown_counter(self):
        rows=[self.row(i,likes=0) for i in range(6)]
        self.assertEqual(I.snapshot(rows,self.now)['groups'][0]['score'],0)

    def test_shares_grow_only_after_real_observation_interval(self):
        rows=[self.row(i,age=100) for i in range(3)]
        for row in rows:
            row.update(share_rate=5,share_history=[dict(observed='2026-11-24T00:00:00Z',shares=2),dict(observed='2026-11-24T00:10:00Z',shares=3)])
        self.assertTrue(all(g['current']==0 for g in I.snapshot(rows,self.now)['groups']))
        for row in rows:row['share_history'][-1]['observed']='2026-11-24T01:00:00Z'
        food=next(g for g in I.snapshot(rows,self.now)['groups'] if g['kind']=='food')
        self.assertEqual(food['status'],'증가 확인');self.assertEqual(food['growing'],3)

    def test_high_relative_response_becomes_candidate_with_evidence(self):
        rows=[self.row(i,likes=10+i,name='말차' if i>=7 else '모찌') for i in range(10)]
        result=I.snapshot(rows,self.now);food=next(g for g in result['groups'] if g['kind']=='food' and g['label']=='말차')
        self.assertEqual(food['status'],'최근 인기 후보');self.assertEqual(food['score'],90)
        self.assertEqual(len(food['current_ids']),3);self.assertEqual(len(food['evidence']),3);self.assertIn('shares',food['evidence'][0])
        rows[-1]['caption']='바뀐 본문';self.assertEqual(I.snapshot(rows,self.now)['tagged'],9)

    def test_update_retries_invalid_json_and_missing_ai_does_not_block_local_results(self):
        from mirror import consumer as C
        import pandas as pd
        row=self.row()
        with D.db() as c:c.execute('delete from trend_tags')
        response=json.dumps(dict(items=[dict(id=row['id'],entities=[dict(name='말차',evidence='말차')],genre='떡·모찌',themes=[],summary='시음')]))
        with patch.object(D,'load',return_value=pd.DataFrame([row])),patch.object(C,'GEMINI_KEY','test'),patch.object(C,'chat',side_effect=['bad json',response]) as chat:
            self.assertIn('1건 해석',I.update(T.CAMPAIGN));self.assertEqual(chat.call_count,2)
        row['caption']='새 말차 제품'
        with patch.object(D,'load',return_value=pd.DataFrame([row])),patch.object(C,'GEMINI_KEY',''),patch.object(C,'chat') as chat:
            self.assertIn('설정 없음',I.update(T.CAMPAIGN));chat.assert_not_called()
        self.assertEqual(I.snapshot([row],self.now)['run']['status'],'error')

    def test_food_discovery_context_allows_new_names_without_accepting_unclassified_posts(self):
        row=dict(caption='한정선에 갔다 왔는데 오픈 전부터 기다려야 했어요',discovery_food=True)
        self.assertTrue(FS.candidate(row));self.assertFalse(FS.accepted(row))
        self.assertFalse(FS.candidate(dict(caption='주가와 목표주가에 대해 분석한 주식시장 현황입니다',discovery_food=True)))


if __name__=='__main__':unittest.main()
