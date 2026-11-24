import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from mirror import decode as D, food_scope as F, trend_social as S, trends as T, trend_topics as B


class FoodSocialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(D,'DB',Path(self.tmp.name)/'decode.db'),
                        patch.object(D.M,'DB',str(Path(self.tmp.name)/'media.db')),
                        patch.dict(os.environ,{},clear=True), patch('mirror.consumer._dotenv')]
        for p in self.patches:p.start()
        self.cfg=T.init()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_food_candidates_reject_false_substrings_and_hashtags(self):
        for text in ['칠성사이다 제로 시음', '편의점 신상 과자', '컵라면 레시피', '라면 가격', '새로 소주 후기', '클라우드 맥주', '하이볼 만드는 법']:
            self.assertTrue(F.candidate({'caption':text}),text)
        for text in ['하프오크 종족이라면 동료를 영입 못한다', '사이다 발언으로 정치판 들썩', '게임 이야기가 주류입니다', '클라우드 서버 신제품', '새로 시작하는 게임', '롯데칠성 목표주가 상향 증권사', '#coffee #커피', '# coffee # 커피', '게임 대회 #칠성사이다']:
            self.assertFalse(F.candidate({'caption':text}),text)
        self.assertEqual(F.brand_group({'caption':'클라우드 맥주 시음'}),'롯데칠성·취급 브랜드')
        self.assertEqual(F.brand_group({'caption':'클라우드 서버 소개'}),'경쟁·카테고리 트렌드')
        self.assertFalse(F.accepted({'food_category':'음료','food_p':.7}))
        self.assertTrue(F.accepted({'food_category':'음료','food_p':.8}))
        self.assertEqual(F.category({'caption':'새로 제로슈거 소주 출시'},'음료'),'주류')
        self.assertEqual(F.category({'caption':'밀키스 신제품 출시'},'식료품'),'음료')

    def test_scope_gate_and_stale_decisions_never_enter_graph_or_topics(self):
        rows=[dict(id='a',caption='칠성사이다 제로 시음',source='YouTube'),dict(id='b',caption='게토레이 광고 출연 선수 경기 결과',source='YouTube'),dict(id='c',caption='음료 이야기 근거 부족',source='YouTube')]
        D.save_rows(T.CAMPAIGN,rows,'youtube')
        def answer(state,q):
            caption=json.loads(state)['caption'];p=.95 if caption.startswith('칠성') else .1 if caption.startswith('게토') else .6
            return {'answers':{'format':{'choice':D.NONE},'food':{'choice':'음료'},'food_scope':{'choice':'식음료' if p>.5 else '기타','probabilities':{'식음료':p,'기타':1-p}}}}
        agent=Mock();agent.predict.side_effect=answer
        D.classify(T.CAMPAIGN,self.cfg,agent=agent)
        s=T.snapshot();self.assertEqual(s['total'],1);self.assertEqual(s['excluded'],2)
        self.assertEqual(len(B.documents(T.CAMPAIGN)),1)
        D.edit_words(s['items'][0]['id'],'본문 수정','')
        s=T.snapshot();self.assertEqual(s['total'],0);self.assertEqual(s['pending'],1)
        self.assertEqual(len(B.documents(T.CAMPAIGN)),0)

    def test_sync_preserves_other_campaigns_and_history(self):
        cid=D.configure('원본','브랜드',[],[],{'후기':'리뷰','기타':'기타'})
        D.save_rows(cid,[dict(id='food',caption='맥주 시음'),dict(id='game',caption='게임 경기 결과')],'threads')
        T.sync();T.sync()
        self.assertEqual(len(D.records(T.CAMPAIGN)),1)
        self.assertEqual(len(D.records(cid)),2)
        with D.db() as c:self.assertEqual(c.execute('select count(*) from snapshots').fetchone()[0],2)

    def test_four_channels_are_enabled_without_credentials_but_not_pretend_live(self):
        jobs=T.source_jobs()
        self.assertEqual({k for k in jobs if k.startswith('social:')},{'social:'+p for p in S.PLATFORMS})
        self.assertTrue(jobs['social:youtube'][1])
        self.assertTrue(all(jobs['social:'+p][1] for p in ['instagram','tiktok','threads']))
        s=T.snapshot();self.assertEqual(len(s['channels']),4)
        self.assertTrue(all(c['status']=='waiting' for c in s['channels'] if c['platform']!='youtube'))
        self.assertNotIn('media:Hacker News',jobs)

    def test_public_url_validation_is_atomic_and_canonical(self):
        cases={'https://youtu.be/abcdefghijk?x=1':('youtube','https://www.youtube.com/watch?v=abcdefghijk'),
               'https://www.instagram.com/reel/abc/?igsh=x':('instagram','https://www.instagram.com/reel/abc/'),
               'https://www.instagram.com/food/reel/abc/?igsh=x':('instagram','https://www.instagram.com/reel/abc/'),
               'https://www.threads.com/@/post/xyz':('threads','https://www.threads.com/@/post/xyz/'),
               'https://threads.net/@a/post/xyz?x=1':('threads','https://www.threads.com/@a/post/xyz/'),
               'https://www.tiktok.com/@a/video/123':('tiktok','https://www.tiktok.com/@a/video/123/')}
        for url,expected in cases.items():self.assertEqual(S.canonical(url),expected)
        for url in ['https://127.0.0.1', 'https://instagram.com.evil.test/reel/x','https://user:pass@instagram.com/reel/x','http://youtube.com/watch?v=abcdefghijk','https://instagram.com/explore/tags/food','https://www.tiktok.com:444/@a/video/123']:
            with self.assertRaises(ValueError):S.canonical(url)
        S.init()
        with self.assertRaises(ValueError):S.add_urls([next(iter(cases)),'https://evil.test'])
        self.assertEqual(S.links('youtube'),[])
        S.add_urls([list(cases)[1]]);self.assertTrue(S.connection('instagram')[0])
        self.assertIn('계정 불필요',S.connection('instagram')[1])

    def test_api_and_public_link_share_identity_without_invented_metrics(self):
        a=S.identity(dict(id='123',url='https://www.instagram.com/reel/ABC/?igsh=x',caption='맥주 시음'),'instagram')
        b=S.identity(dict(id='ABC',url='https://www.instagram.com/reel/ABC/'),'instagram')
        self.assertEqual(a['id'],b['id'])
        row=S.video_row(dict(id='abc',title='커피',view_count=0,repost_count=4),'youtube')
        self.assertEqual(row['views'],0);self.assertIsNone(row['likes']);self.assertIsNone(row['shares']);self.assertEqual(row['metrics']['reposts'],4)

    def test_instagram_uses_eight_fixed_tags_and_only_available_metrics(self):
        def graph(path,**kwargs):
            if path=='ig_hashtag_search':return {'data':[{'id':'tag'}]}
            self.assertNotIn('view_count',kwargs['fields'])
            return {'data':[{'id':'1','caption':'맥주 시음','permalink':'https://www.instagram.com/reel/ABC/','like_count':0}]}
        with patch.dict(os.environ,INSTAGRAM_USER_ID='user'),patch.object(S,'graph',side_effect=graph) as g:
            rows=list(S.instagram())
        self.assertEqual(len(rows),16);self.assertEqual(len({call.kwargs['q'] for call in g.call_args_list if call.args[0]=='ig_hashtag_search'}),8)
        self.assertIsNone(rows[0].get('views'));self.assertEqual(rows[0]['likes'],0)

    def test_threads_keyword_and_tiktok_authorized_only_contracts(self):
        def api(url,token,payload=None):
            if 'threads.com' in url:
                self.assertIn('keyword_search?',url);self.assertIn('search_type=RECENT',url)
                return {'data':[{'id':'t','text':'제로음료 후기','permalink':'https://www.threads.com/@u/post/t/'}]}
            self.assertIn('/v2/video/list/',url);self.assertEqual(payload,{'max_count':20})
            return {'data':{'videos':[{'id':'k','video_description':'맥주','view_count':0,'like_count':3,'share_count':1}]}}
        with patch.dict(os.environ,THREADS_ACCESS_TOKEN='test',TIKTOK_ACCESS_TOKEN='test'),patch.object(S,'api',side_effect=api):
            rows=list(S.threads());self.assertEqual(len(rows),4);self.assertNotIn('views',rows[0])
            row=list(S.tiktok())[0];self.assertEqual((row['views'],row['likes'],row['shares']),(0,3,1))

    def test_threads_public_page_must_match_the_requested_post(self):
        url='https://www.threads.com/@u/post/ABC/'
        post=dict(code='ABC',user={'username':'u'},caption={'text':'커피 신제품 시음'},like_count=0,text_post_app_info={'direct_reply_count':2})
        with patch.object(S.Public,'get',return_value='<html>로그인</html>'),self.assertRaises(RuntimeError):S.public_post(url)
        html='<script type="application/json">'+json.dumps(post)+'</script>'
        with patch.object(S.Public,'get',return_value=html):row=S.public_post(url)
        self.assertEqual(row['id'],'ABC');self.assertIsNone(row['views']);self.assertEqual(row['likes'],0)

    def test_partial_collection_keeps_rows_and_records_public_failures(self):
        def source():
            yield dict(id='a',caption='맥주 시음')
            yield dict(id='b',caption='게임 결과')
            raise RuntimeError('공식 API 사용량 제한')
        S.add_urls(['https://www.tiktok.com/@a/video/123'])
        with patch.dict(os.environ,TIKTOK_ACCESS_TOKEN='test'),patch.object(S,'tiktok',side_effect=source),patch.object(S,'public_post',side_effect=RuntimeError('접근 제한')),patch.object(S,'public_collect',return_value=([],[])):
            rows,errors=S.collect('tiktok')
        self.assertEqual(len(rows),1);self.assertEqual(len(errors),2)
        self.assertEqual(S.links('tiktok')[0]['status'],'error')

    def test_youtube_public_search_retains_flat_views_when_details_fail(self):
        def extract(target,flat=False):
            if flat:return {'entries':[dict(id='abcdefghijk',title='제로 음료 리뷰',view_count=500)]}
            raise RuntimeError('접근 제한')
        with patch.object(S,'ytdlp',side_effect=extract):rows=list(S.youtube())
        self.assertEqual(len(rows),4);self.assertTrue(all(r['views']==500 and r['likes'] is None for r in rows))

    def test_manual_instagram_detail_keeps_current_reel_metric(self):
        url='https://www.instagram.com/reel/ABC/'
        S.add_urls([url])
        grid=dict(id='ABC',url=url,caption='칠성사이다 제로 시음',views=100,view_metric='plays',collection_method='공개 릴스 + 원문',evidence_url='https://www.instagram.com/food/reels/')
        detail=grid|{'views':None,'view_metric':'views','collection_method':'공개 원문 JSON'}
        with patch.object(S,'public_collect',return_value=([grid],[])),patch.object(S,'public_post',return_value=detail):rows,errors=S.collect('instagram')
        self.assertFalse(errors);self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['views'],rows[0]['view_metric']),(100,'plays'))


if __name__=='__main__':unittest.main()
