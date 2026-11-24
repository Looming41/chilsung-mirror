import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from mirror import decode as D, trend_public as P, trend_social as S, trend_discovery as Q


def html(value, script_id=None):
    return '<script type="application/json"' + (f' id="{script_id}"' if script_id else '') + '>' + json.dumps(value) + '</script>'


class PublicSocialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = patch.object(D, 'DB', Path(self.tmp.name) / 'decode.db')
        self.db.start()
        with D.db() as c:
            c.execute("insert into campaigns values('test','test','{}')")

    def tearDown(self):
        self.db.stop()
        self.tmp.cleanup()

    def test_instagram_exact_identity_hidden_counts_and_reels_shape(self):
        detail = dict(code='ABC', media_type=2, product_type='clips', caption={'text':'칠성사이다 제로 시음'}, taken_at=1795421727, like_count=0, comment_count=2)
        reel = dict(code='ABC', __typename='XIGPolarisVideoMedia', product_type='clips', play_count=3877, like_count=0, comment_count=2)
        profile = {'xig_user_by_username':dict(username='food', is_private=False, polaris_clips_connection={'edges':[{'node':reel}]})}
        row = P.instagram(html(profile))[0]
        self.assertEqual((row['views'], row['view_metric'], row['creator']), (3877, 'plays', 'food'))
        self.assertIsNone(row['caption'])
        row = P.instagram(html([detail,detail|{'code':'OTHER','like_count':999}]), 'ABC')[0]
        self.assertEqual((row['id'], row['likes'], row['comments']), ('ABC', 0, 2))
        self.assertIsNone(row['views'])
        hidden = P.instagram(html(detail|{'play_count':900,'like_and_view_counts_disabled':True}), 'ABC')[0]
        self.assertIsNone(hidden['likes']); self.assertIsNone(hidden['views'])
        profile['xig_user_by_username']['is_private'] = True
        with self.assertRaises(RuntimeError): P.instagram(html(profile))
        with self.assertRaises(RuntimeError): P.instagram(html(detail), 'MISSING')

    def test_tiktok_embed_discovery_and_exact_detail(self):
        key = '7709475091704352020'
        embed = {'source':{'data':{'/embed/@food':{'videoList':[{'id':key,'desc':'어묵 요리'},{'id':'2','privateItem':True}]}}}}
        rows = P.tiktok(html(embed,'__FRONTITY_CONNECT_STATE__'), handle='food')
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]['id'], key)
        item = dict(id=key,author={'uniqueId':'food'},createTime='1795000000',desc='어묵 요리',stats={'playCount':999,'diggCount':3},statsV2={'playCount':'150000','diggCount':'0','commentCount':'57','shareCount':'192'})
        payload = {'__DEFAULT_SCOPE__':{'webapp.video-detail':{'statusCode':0,'itemInfo':{'itemStruct':item}}}}
        row = D.normalize(P.tiktok(html(payload,'__UNIVERSAL_DATA_FOR_REHYDRATION__'),key)[0], 'tiktok')
        self.assertEqual((row['views'],row['likes'],row['comments'],row['shares']), (150000,0,57,192))
        with self.assertRaises(RuntimeError): P.tiktok(html(payload,'__UNIVERSAL_DATA_FOR_REHYDRATION__'),'999')
        with self.assertRaises(RuntimeError): P.tiktok('<html>로그인</html>',key)

    def test_threads_search_excludes_suggestions_and_flattens_matching_replies(self):
        post = dict(code='ABC', user={'username':'food'}, caption={'text':'하이볼 시음'}, like_count=50, text_post_app_info={'direct_reply_count':73,'repost_count':2,'reshare_count':5})
        other = post|{'code':'UNRELATED'}
        payload = {'suggestions':other,'searchResults':{'edges':[{'node':{'thread':{'thread_items':[{'post':post},{'post':post|{'code':'REPLY'}}]}}}]}}
        rows = P.threads(html(payload),search=True)
        self.assertEqual({r['id'] for r in rows},{'ABC','REPLY'})
        row = rows[0]
        self.assertEqual((row['likes'],row['comments'],row['metrics']['reposts'],row['shares']),(50,73,2,5))
        self.assertIsNone(row['views'])
        self.assertEqual(len(P.threads(html(payload),'ABC')),1)
        with self.assertRaises(RuntimeError): P.threads(html(payload),'DELETED')
        with self.assertRaises(RuntimeError): P.threads(html({'home':post}),search=True)

    def test_public_http_rejects_untrusted_hosts_before_request(self):
        for url in ['https://127.0.0.1','https://instagram.com.evil.org/','http://threads.com','https://user:secret@tiktok.com','https://www.tiktok.com:444/']:
            with patch.object(P.urllib.request,'build_opener') as opener,self.assertRaises(ValueError): P.get(url)
            opener.assert_not_called()

    def test_instagram_keyword_collection_preserves_native_plays_and_partial_rows(self):
        detail = dict(code='ABC',media_type=2,product_type='clips',caption={'text':'칠성사이다 제로 시음'},taken_at=1795421727)
        reels = [dict(code=key,__typename='XIGPolarisVideoMedia',play_count=1234,caption={'text':'칠성사이다 제로 시음'},user={'username':key}) for key in ['ABC','BROKEN']]
        def get(url):
            if '/BROKEN/' in url: raise RuntimeError('접근 제한')
            if '/reel/' in url: return html(detail)
            return html({'xig_logged_out_popular_search_media_info':{'edges':[{'node':r} for r in reels]}, 'unrelated':detail|{'code':'NO'}})
        Q.init()
        with patch.object(Q,'due',return_value=[{'query':'제로 음료'}]),patch.object(P,'get',side_effect=get): rows,errors=Q.collect('instagram')
        self.assertEqual(len(rows),2); self.assertTrue(errors)
        self.assertEqual({r['id'] for r in rows},{'ABC','BROKEN'})
        self.assertTrue(all(r['views']==1234 and r['discovery_food'] for r in rows))
        partial=next(r for r in rows if r['id']=='BROKEN')
        self.assertIsNone(partial['likes']);self.assertIsNone(partial['published'])
        D.save_rows('test',rows,'instagram')
        self.assertEqual({r['collection_method'] for r in D.records('test')},{'공개 인기 키워드 + 원문','공개 인기 키워드 미리보기'})
        self.assertEqual(Q.snapshot()['totals'][0]['creators'],2)

    def test_threads_does_not_count_nested_quoted_posts(self):
        quoted=dict(code='QUOTE',user={'username':'quoted'},caption={'text':'무관한 인용'})
        post=dict(code='MATCH',user={'username':'food'},caption={'text':'말차 시음'},text_post_app_info={'share_info':{'quoted_post':quoted}})
        rows=P.threads(html({'searchResults':{'edges':[{'node':{'thread':{'thread_items':[{'post':post}]}}}]}}),search=True)
        self.assertEqual([r['id'] for r in rows],['MATCH'])

    def test_gzip_is_decoded_and_expanded_size_is_bounded(self):
        import gzip
        response=Mock(url='https://www.threads.com/search?q=food',headers={'Content-Encoding':'gzip'})
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        with patch.object(P.urllib.request,'build_opener') as opener:
            opener.return_value.open.return_value=response
            response.read.return_value=gzip.compress('공개 검색'.encode())
            self.assertEqual(P.get(response.url),'공개 검색')
            response.read.return_value=gzip.compress(b'x'*8_000_001)
            with self.assertRaisesRegex(RuntimeError,'압축 해제'):P.get(response.url)

    def test_keyword_links_only_accept_actual_tiktok_video_anchors(self):
        links='<a href="https://www.tiktok.com/@food/video/1234?x=1&amp;y=2">one</a><a href="https://tiktok.com/@food/video/1234">duplicate</a>'
        links+='<a href="https://tiktok.com.evil.org/@food/video/999">bad</a><a href="https://u:p@tiktok.com/@food/video/888">bad</a>'
        links+='<a href="http://tiktok.com/@food/video/777">bad</a><script>"https://www.tiktok.com/@food/video/666"</script>'
        self.assertEqual([r['id'] for r in Q.tiktok_links(links)],['1234'])

    def test_discovery_queue_rotates_and_reserves_slots_for_learned_foods(self):
        import time
        first=Q.due('instagram');self.assertEqual(len(first),8)
        with D.db() as c:
            c.executemany('update trend_queries set last_run=? where platform=? and query=?',[(time.time(),'instagram',r['query']) for r in first])
        Q.learn(['복숭아 하이볼','밀크 모찌','https://bad.test','1'],'공개 인기 검색 연관어')
        second=Q.due('instagram');self.assertFalse({r['query'] for r in first}&{r['query'] for r in second})
        self.assertEqual(sum(r['origin']!='초기 식음료 탐색어' for r in second),2)
        self.assertEqual(len(second),8)
        self.assertEqual({r['query'] for r in second[-2:]},{'복숭아 하이볼','밀크 모찌'})

    def test_reaction_velocity_is_separate_and_missing_counters_stay_missing(self):
        row = dict(id='food',caption='커피 시음',likes=10,shares=4)
        D.save_rows('test',[row],'threads','a','2026-11-24T00:00:00Z')
        D.save_rows('test',[row|{'likes':30,'shares':10}],'threads','b','2026-11-24T02:00:00Z')
        cfg={'brand':'','formats':{'리뷰':'리뷰'}}
        result=D.load('test',cfg).iloc[0]
        self.assertEqual(result.share_rate,3);self.assertEqual(result.reaction_rate,10); self.assertIsNone(result.rate); self.assertIsNone(result.views)
        D.save_rows('test',[row|{'likes':None}],'threads','c','2026-11-24T03:00:00Z')
        self.assertIsNone(D.load('test',cfg).iloc[0].reaction_rate)


if __name__ == '__main__': unittest.main()
