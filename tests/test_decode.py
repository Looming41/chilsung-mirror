import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import sys
from types import SimpleNamespace

import pandas as pd

from mirror import decode as D


class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = patch.object(D, "DB", Path(self.tmp.name) / "decode.db")
        self.db.start()
        self.media_db = patch.object(D.M, "DB", str(Path(self.tmp.name) / "media.db"))
        self.media_db.start()
        self.cid = D.configure("테스트 캠페인", "칠성사이다", ["format:인벤", "media:블루스카이"], [], {"비교": "두 음료 비교", "후기": "시음 후기"})
        self.cfg = D.campaigns()[0]["config"]

    def tearDown(self):
        self.media_db.stop()
        self.db.stop()
        self.tmp.cleanup()

    def row(self, **kwargs):
        return dict(id="123", caption="칠성사이다 시음", ownerUsername="creator_a", videoPlayCount=100,
                    likesCount=0, url="https://www.instagram.com/reel/abc/", **kwargs)

    def test_missing_zero_and_private_metrics(self):
        r = D.normalize(dict(id="1", videoPlayCount=0, videoViewCount=700, likesCount=-1), "instagram")
        self.assertEqual((r["views"], r["view_metric"], r["likes"], r["shares"]), (0, "plays", None, None))
        self.assertEqual(D.number("1.2만"), 12000)
        self.assertIsNone(D.number("NaN"))
        self.assertIsNone(D.number(True))

    def test_only_existing_non_apify_sources(self):
        self.assertFalse(hasattr(D.M, "apify"))
        self.assertFalse(hasattr(D.M, "APIFY"))
        with self.assertRaises(ValueError):
            D.configure("bad", "brand", ["media:인스타 릴스"], [], self.cfg["formats"])

    def test_snapshot_idempotency_and_rate(self):
        D.save_rows(self.cid, [self.row()], "instagram", "a", "2026-11-24T00:00:00Z")
        changed = self.row() | {"videoPlayCount": 340}
        D.save_rows(self.cid, [changed], "instagram", "b", "2026-11-24T02:00:00Z")
        D.save_rows(self.cid, [self.row()], "instagram", "a", "2026-11-24T03:00:00Z")
        d = D.load(self.cid, self.cfg)
        self.assertEqual(len(d), 1)
        self.assertEqual(d.iloc[0]["views"], 340)
        self.assertEqual(d.iloc[0]["rate"], 120)
        with D.db() as c:
            self.assertEqual(c.execute("select count(*) from snapshots").fetchone()[0], 2)

    def test_counter_decrease_not_growth(self):
        D.save_rows(self.cid, [self.row()], "instagram", "a", "2026-11-24T00:00:00Z")
        D.save_rows(self.cid, [self.row() | {"videoPlayCount": 50}], "instagram", "b", "2026-11-24T02:00:00Z")
        self.assertIsNone(D.load(self.cid, self.cfg).iloc[0]["rate"])

    def test_platform_identity_and_bad_rows(self):
        n, errors = D.save_rows(self.cid, [self.row(), {"error": "private"}], "instagram")
        self.assertEqual((n, len(errors)), (1, 1))
        D.save_rows(self.cid, [self.row()], "tiktok")
        self.assertEqual(len(D.records(self.cid)), 2)

    def test_classification_uses_all_text_and_invalidates(self):
        D.save_rows(self.cid, [self.row(transcript="대본의 칠성사이다", on_screen_text="화면 속 비교")], "instagram")
        agent = Mock()
        agent.predict.return_value = {"answers": {"format": {"choice": "비교", "probabilities": {"비교": .9}}, "brand": {"choice": "확인", "probabilities": {"확인": .8}}}}
        module = SimpleNamespace(load=Mock(return_value=agent))
        with patch.dict(sys.modules, laya=module):
            self.assertEqual(D.classify(self.cid, self.cfg), 1)
            state = json.loads(agent.predict.call_args.args[0])
            self.assertEqual(state["on_screen_text"], "화면 속 비교")
            self.assertEqual(state["transcript"], "대본의 칠성사이다")
            self.assertEqual(D.classify(self.cid, self.cfg), 0)
            self.assertEqual(module.load.call_count, 1)
            d = D.load(self.cid, self.cfg)
            self.assertEqual(d.iloc[0]["fmt"], "비교")
            D.edit_words(d.iloc[0]["id"], "다른 대본", "다른 자막")
            self.assertIsNone(D.load(self.cid, self.cfg).iloc[0]["fmt"])
            self.assertEqual(D.classify(self.cid, self.cfg), 1)
            self.assertIsNone(D.load(self.cid, self.cfg | {"brand": "밀키스"}).iloc[0]["fmt"])

    def test_invalid_decision_not_saved(self):
        D.save_rows(self.cid, [self.row()], "instagram")
        agent = Mock()
        agent.predict.return_value = {"answers": {"format": {"choice": "지어낸 포맷"}}}
        with patch.dict(sys.modules, laya=SimpleNamespace(load=Mock(return_value=agent))), self.assertRaises(ValueError):
            D.classify(self.cid, self.cfg)
        with D.db() as c:
            self.assertEqual(c.execute("select count(*) from decisions").fetchone()[0], 0)

    def test_median_and_small_sample(self):
        d = pd.DataFrame({"views": [10] * 5 + [100] * 4 + [100000], "fmt": ["A"] * 5 + ["B"] * 4 + ["C"]})
        r = D.ranking(d).set_index("포맷")
        self.assertEqual(r.loc["B", "중앙 조회수"], 100)
        self.assertEqual(r.loc["B", "판정"], "표본 부족 (<5)")
        self.assertAlmostEqual(r.loc["A", "캠페인 대비"], 10 / 55)
        self.assertTrue(pd.isna(D.ranking(d.assign(views=0)).iloc[0]["캠페인 대비"]))

    def test_import_keeps_enrichment(self):
        D.save_rows(self.cid, [self.row(transcript="기존 대본")], "instagram")
        D.save_rows(self.cid, [self.row()], "instagram")
        self.assertEqual(D.records(self.cid)[0]["transcript"], "기존 대본")

    def seed_existing(self):
        D.F.save([dict(key="one", lane="커뮤니티", grp="인벤", title="칠성사이다 리뷰", url="https://example.com/one", views=100, likes=5)])
        c = D.M.db()
        c.execute("update content set last_seen='2026-11-24T00:00:00+00:00'")
        for key, url in [("same", "https://example.com/one"), ("other", "https://example.com/two")]:
            c.execute("insert into items(id,source,kind,text,url,last_seen,engagement) values(?,?,?,?,?,?,?)",
                      (key, "블루스카이", "SNS", "칠성사이다 후기", url, "2026-11-24T00:00:00+00:00", 999))
        c.commit()
        c.close()

    def test_existing_bridge_preserves_metrics_and_observation(self):
        self.seed_existing()
        with patch.object(D.M, "collect") as fetch:
            self.assertEqual(D.collect(self.cid, self.cfg), (2, []))
            self.assertEqual(D.collect(self.cid, self.cfg), (2, []))
            fetch.assert_not_called()
        d = D.load(self.cid, self.cfg)
        self.assertEqual(d[d.platform == "community"].views.iloc[0], 100)
        self.assertTrue(d[d.platform == "social"].views.isna().all())
        self.assertTrue(d.rate.isna().all())
        with D.db() as c:
            self.assertEqual(c.execute("select count(*) from snapshots").fetchone()[0], 2)
        c = D.M.db()
        c.execute("update content set views=340,last_seen='2026-11-24T02:00:00+00:00'")
        c.commit();c.close()
        D.collect(self.cid, self.cfg)
        self.assertEqual(D.load(self.cid, self.cfg).query("platform == 'community'").rate.iloc[0], 120)

    def test_keywords_and_selected_sources(self):
        self.seed_existing()
        cfg = self.cfg | {"sources": ["media:블루스카이"], "keywords": ["없는단어"]}
        self.assertEqual(D.collect(self.cid, cfg), (0, []))
        self.assertEqual(D.collect(self.cid, cfg | {"keywords": ["후기"]}), (2, []))
        self.assertTrue(all(r["source"] == "블루스카이" for r in D.records(self.cid)))

    def test_refresh_reuses_only_selected_collectors(self):
        self.seed_existing()
        with patch.object(D.F, "community", return_value={"인벤": "1건"}) as board, patch.object(D.M, "collect") as media:
            D.collect(self.cid, self.cfg, refresh=True)
        board.assert_called_once_with(names=["인벤"])
        media.assert_called_once_with(only_sources=["블루스카이"])

    def test_media_collector_honors_source_filter(self):
        free, paid = Mock(return_value=[]), Mock(return_value=[])
        with patch.object(D.M, "sources", return_value={"free": ("SNS", free, True, ""), "paid": ("SNS", paid, True, "")}):
            D.M.collect(only_sources=["free"])
        free.assert_called_once()
        paid.assert_not_called()

    def test_community_collector_honors_source_filter(self):
        with patch.object(D.M, "get", return_value="") as get:
            result = D.F.community(names=["인벤"])
        self.assertEqual(set(result), {"인벤"})
        get.assert_called_once()
        self.assertIn("inven.co.kr", get.call_args.args[0])

    def test_legacy_campaign_keeps_data_but_needs_source_selection(self):
        with D.db() as c:
            c.execute("update campaigns set config=? where id=?", (json.dumps({"brand": "칠성", "formats": self.cfg["formats"], "mode": "인스타 계정·릴스 URL"}), self.cid))
        self.assertEqual(D.campaigns()[0]["config"]["sources"], [])

    def test_url_validation_and_script_escape(self):
        for url in ["http://example.com/a", "https://127.0.0.1/a", "https://[::1]/", "https://localhost/a", "file:///etc/passwd", "https://user:pass@example.com"]:
            self.assertFalse(D.public_url(url), url)
        with patch.object(D.socket, "getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 443))]), self.assertRaises(ValueError):
            D.download("https://example.com/a", Path(self.tmp.name) / "file")
        html = D.chart_html(pd.DataFrame([{"caption": "</script><script>alert(1)</script>", "views": 1, "likes": 1}]))
        self.assertNotIn("</script><script>alert", html)
        encoded = html.split("const data=", 1)[1].split(";\n", 1)[0]
        self.assertEqual(json.loads(encoded)[0]["caption"], "</script><script>alert(1)</script>")

    def test_csv_external_text_is_not_a_formula(self):
        d = pd.DataFrame([dict(caption=' =HYPERLINK("bad")', views=0, video_url="signed-secret-url")])
        exported = pd.read_csv(io.BytesIO(D.csv_bytes(d)))
        self.assertTrue(exported.caption.iloc[0].startswith("'"))
        self.assertEqual(exported.views.iloc[0], 0)
        self.assertNotIn("video_url", exported.columns)


if __name__ == "__main__":
    unittest.main()
