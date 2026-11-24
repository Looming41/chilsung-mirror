import fcntl
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

import numpy as np
from mirror import decode as D, trends as T, trend_topics as B


class TrendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(D, 'DB', Path(self.tmp.name) / 'decode.db'), patch.object(D.M, 'DB', str(Path(self.tmp.name) / 'media.db'))]
        for p in self.patches:p.start()
        self.cfg = T.init()

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def seed(self, n=12):
        now=D.now()
        rows=[dict(id=str(i), caption='칠성사이다 음료 리뷰 ' + str(i), source='test', url=f'https://example.org/{i}', views=100+i, first_seen=now, published=now) for i in range(n)]
        D.save_rows(T.CAMPAIGN, rows, 'community', batch='first', observed=now)
        self.decide()
        return rows

    def decide(self):
        agent=Mock()
        agent.predict.side_effect=lambda *a, **kw: {'answers':{'format':{'choice':D.NONE},'food':{'choice':'음료'},'food_scope':{'choice':'식음료','probabilities':{'식음료':.95,'기타':.05}}}}
        D.classify(T.CAMPAIGN,self.cfg,agent=agent)

    def test_latest_payload_and_chronological_growth(self):
        row=dict(id='x', caption='테스트 콘텐츠', source='test', views=100)
        D.save_rows(T.CAMPAIGN,[row],'community','one','2026-11-24T09:00:00+09:00')
        D.save_rows(T.CAMPAIGN,[row|{'views':220}],'community','three','2026-11-24T02:00:00Z')
        D.save_rows(T.CAMPAIGN,[row|{'views':140}],'community','two','2026-11-24T01:00:00Z')
        d=D.load(T.CAMPAIGN,self.cfg).iloc[0]
        self.assertEqual((d.views,d.rate,d.acceleration),(220,80,40))
        self.assertEqual(len(d.history),3)

    def test_identity_preserves_sources_and_same_title_different_urls(self):
        c=D.M.db()
        for source,url in [('A','https://example.org/1'),('B','https://example.org/1'),('A','https://example.org/2')]:
            D.M.save(c,source,'뉴스',[('동일 제목의 기사 테스트',url,None,0)])
        self.assertEqual(c.execute('select count(*) from items').fetchone()[0],3)
        c.close()

    def test_zero_views_and_hidden_youtube_metrics(self):
        D.F.save([dict(key='zero',lane='커뮤니티',grp='인벤',title='조회수 없는 첫 글',views=0)])
        c=D.F.db();self.assertEqual(c.execute('select views from content').fetchone()[0],0);c.close()
        r=D.F.yt_rows([dict(id='yt',snippet=dict(channelId='ch',title='테스트',publishedAt=D.now()),statistics={'viewCount':'0'})])[0]
        self.assertIsNone(r['likes']);self.assertIsNone(r['comments'])

    def test_metric_input_validation(self):
        n,errors=D.save_rows(T.CAMPAIGN,[dict(id='bad',metrics=['likes'])],'social')
        self.assertEqual(n,0);self.assertEqual(len(errors),1)

    def test_worker_singleton_and_cooperative_stop(self):
        lock=(D.DB.parent/'trend-worker.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with patch.object(T,'cycle') as cycle:T.watch(once=True)
        cycle.assert_not_called();lock.close()
        with patch.object(T,'sync') as sync:
            self.assertEqual(T.fetch_one('media:test','test',stopped=lambda:True),(0,0))
        sync.assert_not_called()

    def test_start_does_not_duplicate_live_worker(self):
        with D.db() as c:c.execute("update trend_worker set heartbeat=?,status='waiting'",(time.time(),))
        with patch.object(T.subprocess,'Popen') as popen:T.start()
        popen.assert_not_called();self.assertEqual(T.control()['enabled'],1)
        T.stop();self.assertEqual(T.control()['enabled'],0)

    def test_source_failure_is_recorded_and_not_exposed(self):
        with D.db() as c:c.execute("insert into trend_sources(id,label) values('media:test','test')")
        fn=Mock(side_effect=RuntimeError('secret-token=private'))
        with patch.object(D.M,'sources',return_value={'test':('뉴스',fn,True,'')}),patch.object(T,'source_jobs',return_value={'media:test':('test',True,'',600)}):
            T.fetch_one('media:test','test')
        with D.db() as c:r=dict(c.execute('select * from trend_sources').fetchone())
        self.assertEqual(r['status'],'error');self.assertEqual(r['error'],'RuntimeError')
        self.assertGreater(r['next_due'],time.time())

    def test_tiny_corpus_waits_and_stale_worker_visible(self):
        self.seed(3)
        self.assertIn('3/10',B.update(T.CAMPAIGN))
        with D.db() as c:c.execute("update trend_worker set status='running',heartbeat=1")
        s=T.snapshot();self.assertFalse(s['alive']);self.assertEqual(s['visible'],3)
        self.assertEqual(s['rising'],0)

    def test_topic_cache_atomicity_and_text_invalidation(self):
        self.seed()
        vectors=np.vstack([np.eye(2)[i//6] for i in range(12)]).astype('float32')
        embedder=Mock();embedder.embed.return_value=vectors
        labels=np.array([0]*6+[1]*6)
        with patch.object(B,'embedder',return_value=embedder),patch.object(B,'cluster',return_value=(labels,{0:'야구',1:'경기'})):
            self.assertIn('2개 주제',B.update(T.CAMPAIGN))
            self.assertIn('변경 없음',B.update(T.CAMPAIGN))
        self.assertEqual(embedder.embed.call_count,1)
        s=T.snapshot();self.assertEqual(len(s['topics']),2)
        vid=D.records(T.CAMPAIGN)[0]['id'];D.edit_words(vid,'새로운 본문','')
        self.assertNotIn(vid,[r['id'] for r in T.snapshot()['items']]);self.assertEqual(T.snapshot()['pending'],1)
        self.decide()
        with patch.object(B,'embedder',return_value=embedder),patch.object(B,'cluster',side_effect=RuntimeError('failed')),self.assertRaises(RuntimeError):B.update(T.CAMPAIGN)
        with D.db() as c:self.assertEqual(c.execute('select count(*) from trend_topics').fetchone()[0],2)

    def test_topic_id_matching_is_one_to_one(self):
        previous=[dict(id='a',centroid=json.dumps([1,0]))]
        ids=B.stable_ids({0:np.array([1,0]),1:np.array([.99,.01])},previous)
        self.assertEqual(ids[0],'a');self.assertNotEqual(ids[1],'a')

    def test_bertopic_korean_and_noise(self):
        self.assertNotIn('ㅋㅋ', B.keyword_text('야구 ㅋㅋ https://example.org/test'))
        self.assertNotIn('example', B.keyword_text('야구 https://example.org/test'))
        texts=['야구 선수 경기 승리 기록']*6+['사이다 음료 신제품 출시']*6
        rng=np.random.default_rng(7)
        vectors=np.vstack([np.array([1,0])+rng.normal(0,.01,2) for _ in range(6)]+[np.array([0,1])+rng.normal(0,.01,2) for _ in range(6)])
        labels,names=B.cluster(texts,vectors)
        self.assertEqual(len(names),2)
        self.assertTrue(any('야구' in v or '선수' in v for v in names.values()))
        self.assertTrue(any('음료' in v or '사이다' in v for v in names.values()))

    def test_shared_trend_url_does_not_drop_different_keywords(self):
        c=D.M.db()
        D.M.save(c,'구글 트렌드 KR','트렌드',[('제로 음료','https://trends.google.com/trending?geo=KR',None,0),('라면 신제품','https://trends.google.com/trending?geo=KR',None,0)])
        c.close();T.sync()
        self.assertEqual(len(D.records(T.CAMPAIGN)),2)

    def test_imported_campaign_content_joins_global_once(self):
        cid=D.configure('외부 파일','브랜드',[],[],{'후기':'후기','뉴스':'뉴스'})
        D.save_rows(cid,[dict(id='import1',caption='업로드한 칠성사이다 리뷰')],'instagram')
        T.sync();T.sync()
        self.assertEqual(len(D.records(T.CAMPAIGN)),1)

    def test_pipeline_orders_local_analysis_and_reuses_worker(self):
        with patch.object(T,'classifier',return_value=object()),patch.object(D,'classify',return_value=2) as classify,patch.object(B,'update',return_value='주제 완료') as topics:
            T.watch(once=True,collect_sources=False)
        classify.assert_called_once();topics.assert_called_once_with(T.CAMPAIGN,tick=T.stage)
        with D.db() as c:
            kinds=[r[0] for r in c.execute('select kind from trend_events order by id')]
            self.assertLess(kinds.index('classified'),kinds.index('topics'))
            self.assertEqual(c.execute('select status from trend_worker').fetchone()[0],'stopped')

    def test_escape_dashboard_payload(self):
        html=T.html(dict(value='</script><script>alert(1)</script>'))
        self.assertNotIn('</script><script>alert(1)',html)
        self.assertIn('\\u003c/script>',html)


if __name__=='__main__':unittest.main()
