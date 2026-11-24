"""실험 전용 추론 호출. 병렬 요청의 최대 비용도 먼저 예약한다."""
import json
import threading
import time

from . import consumer as C, scenes as S

LIMIT = 10.0


class Budget:
    def __init__(self, model, run_id, job):
        if model not in C.PRICES:
            raise ValueError("비용을 계산할 수 없는 모델입니다")
        self.model, self.run_id, self.job = model, run_id, job
        self.records, self.pending, self.spent = [], 0.0, 0.0
        self.lock = threading.RLock()

    def snapshot(self):
        with self.lock:
            return {"limit": LIMIT, "estimated_usd": round(self.spent, 6), "reserved_usd": round(self.pending, 6),
                    "calls": len(self.records), "input_tokens": sum(r["input"] for r in self.records),
                    "output_tokens": sum(r["output"] for r in self.records), "reasoning_tokens": sum(r["reasoning_tokens"] for r in self.records),
                    "uncertain": any(r["status"] == "unknown" for r in self.records), "model": self.model,
                    "basis": "Standard 단가·입력 캐시쓰기 여유 25%·출력에 추론 포함. 실제 게이트웨이 청구와 다를 수 있음", "records": [r.copy() for r in self.records]}

    def publish(self):
        self.job["usage"] = self.snapshot()

    def ask(self, stage, system, prompt, read, *, effort="high", max_output=16384):
        cl = C._clients.get("openai")
        if cl is None:
            raise RuntimeError("OpenAI 연결이 필요합니다")
        ip, op = C.PRICES[self.model]
        for attempt in range(2):
            # UTF-8 바이트 상계와 프레이밍 여유. 긴 문맥 할증 구간에 들어가지 않는다.
            upper = len((system + prompt).encode("utf-8")) + 4096
            if upper > 200_000:
                raise ValueError("요청 자료가 너무 큽니다. 분석 묶음을 줄여야 합니다")
            reserve = (upper * ip * 1.25 + max_output * op) / 1e6
            with self.lock:
                if self.spent + self.pending + reserve > LIMIT:
                    raise RuntimeError("실험 비용 상한 $10 안에서 다음 호출을 보장할 수 없어 중단했습니다. 중간 결과는 저장됩니다")
                self.pending += reserve
                self.publish()
            started, entry = time.monotonic(), None

            def account(r):
                nonlocal entry
                u = r.usage
                tin, tout = (u.input_tokens, u.output_tokens) if u else (0, 0)
                details = getattr(u, "output_tokens_details", None)
                cost = (tin * ip * 1.25 + tout * op) / 1e6 if u else reserve
                entry = {"stage": stage, "model": self.model, "effort": effort, "input": tin, "output": tout,
                         "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                         "usd": cost, "status": r.status if u else "unknown", "valid": False,
                         "seconds": round(time.monotonic() - started, 2)}
                with self.lock:
                    self.pending -= reserve
                    self.spent += cost
                    self.records.append(entry)
                    self.publish()

            try:
                answer, _, _ = C.ask_openai(cl.with_options(timeout=240, max_retries=0), self.model, system, prompt,
                    reasoning={"effort": effort}, max_output_tokens=max_output, service_tier="default",
                    text={"format": {"type": "json_object"}}, on_usage=account,
                    audit={"run": self.run_id, "stage": stage, "effort": effort})
            except Exception as e:
                if entry is None:
                    # 응답 유실도 과금됐을 수 있으므로 예약액을 돌려주지 않는다. HTTP 오류는 재시도하지 않는다.
                    with self.lock:
                        self.pending -= reserve
                        self.spent += reserve
                        self.records.append({"stage": stage, "model": self.model, "effort": effort, "input": 0, "output": 0,
                                             "reasoning_tokens": 0, "usd": reserve, "status": "unknown", "valid": False,
                                             "seconds": round(time.monotonic() - started, 2)})
                        self.publish()
                raise RuntimeError(f"{stage} AI 연결 실패({type(e).__name__}). 자동 재호출 없이 중간 결과를 저장했습니다") from e
            if entry["status"] != "completed":
                raise RuntimeError(f"{stage} AI 응답이 끝나지 않았습니다({entry['status']}). 추론·출력 한도 내 완료하지 못한 답은 사용하지 않습니다")
            try:
                parsed = read(S._json(answer))
                entry["valid"] = True
                self.publish()
                return parsed
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                if attempt:
                    raise ValueError(f"{stage} 응답 검증 실패: {str(e)[:120]}") from e
                # 성공한 HTTP 응답의 JSON 내용만 한 번 교정한다. 교정 호출도 같은 예산에 포함된다.
                prompt += "\n응답 검증 오류: " + str(e)[:120] + "\n스키마와 실제 원문을 확인해 JSON 전체를 다시 작성하세요."
        raise AssertionError("unreachable")


def check():
    from concurrent.futures import ThreadPoolExecutor, wait
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock, patch
    import os

    cl = MagicMock()
    cl.with_options.return_value = cl
    response = lambda **kw: NS(status="completed", output_text='{"ok":true}', usage=NS(input_tokens=100, output_tokens=300, output_tokens_details=NS(reasoning_tokens=250)), **kw)
    with TemporaryDirectory() as tmp, patch.object(C, "LEDGER", os.path.join(tmp, "usage.jsonl")), patch.dict(C._clients, {"openai": cl}):
        cl.responses.create.return_value = response()
        b = Budget("gpt-6-luna", "unit", {})
        assert b.ask("analysis", "JSON", "JSON", lambda x:x)["ok"]
        sent = cl.responses.create.call_args.kwargs
        assert sent["reasoning"] == {"effort":"high"} and sent["max_output_tokens"] == 16384 and sent["service_tier"] == "default"
        assert b.snapshot()["reasoning_tokens"] == 250 and b.pending == 0 and b.spent > 0
        assert json.loads(open(C.LEDGER).readline())["run"] == "unit"
        cl.responses.create.side_effect = [NS(status="completed", output_text='{"wrong":true}', usage=response().usage), response()]
        b = Budget("gpt-6.2-sol", "repair", {})
        assert b.ask("report", "JSON", "JSON", lambda m:m["ok"]) is True and b.snapshot()["calls"] == 2
        assert not b.records[0]["valid"] and b.records[1]["valid"]
        cl.responses.create.side_effect = TimeoutError()
        b = Budget("gpt-6.2-sol", "timeout", {})
        calls = cl.responses.create.call_count
        try:
            b.ask("before", "JSON", "JSON", lambda x:x, max_output=100)
            raise AssertionError("timeout ignored")
        except RuntimeError:
            assert cl.responses.create.call_count == calls+1 and b.snapshot()["uncertain"] and b.pending == 0 and b.spent > 0
        cl.responses.create.side_effect = None
        cl.responses.create.return_value = NS(status="incomplete", output_text='', usage=response().usage)
        b = Budget("gpt-6.2-sol", "incomplete", {})
        try:
            b.ask("review", "JSON", "JSON", lambda x:x)
            raise AssertionError("incomplete accepted")
        except RuntimeError:
            assert b.snapshot()["calls"] == 1 and b.spent > 0
        started, release = threading.Event(), threading.Event()
        def hold(**kw):
            started.set()
            assert release.wait(3)
            return response()
        cl.responses.create.side_effect = hold
        b = Budget("gpt-6.2-sol", "concurrent", {})
        with patch(__name__ + ".LIMIT", .015), ThreadPoolExecutor(3) as pool:
            fs = [pool.submit(b.ask, "before", "JSON", "JSON", lambda x:x, max_output=100) for _ in range(3)]
            assert started.wait(2)
            done, pending = wait(fs, timeout=.2)
            assert len(done) == 2 and all(isinstance(f.exception(), RuntimeError) for f in done)
            assert b.spent+b.pending <= .015
            release.set()
        assert len(b.records) == 1 and b.pending < 1e-9
        calls = cl.responses.create.call_count
        try:
            b.ask("analysis", "JSON", "가"*100000, lambda x:x)
            raise AssertionError("oversized input accepted")
        except ValueError:
            assert cl.responses.create.call_count == calls
    print("budget ok · 추론·출력 제한·실행별 원장·병렬 예약·교정 과금·유실/불완전 응답")


if __name__ == "__main__":
    check()
