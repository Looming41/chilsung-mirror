"""TinyTroupe(Microsoft) 안에서 실행. 인물마다 TinyPerson 을 정의하고 자극을 들려준 뒤 말한 내용을 모은다."""
import json
import os
import sys

src, dst = sys.argv[1:3]
J = json.load(open(src))
open("config.ini", "w").write(f"""[OpenAI]
API_TYPE=openai
MODEL={os.environ['LLM_MODEL']}
REASONING_MODEL={os.environ['LLM_MODEL']}
MAX_COMPLETION_TOKENS=4000
TIMEOUT=600
MAX_ATTEMPTS=3
MAX_CONCURRENT_MODEL_CALLS=1
CACHE_API_CALLS=False
[Simulation]
PARALLEL_AGENT_GENERATION=False
PARALLEL_AGENT_ACTIONS=False
[Cognition]
ENABLE_MEMORY_CONSOLIDATION=False
ENABLE_CONTINUOUS_CONTEXTUAL_SEMANTIC_MEMORY_RETRIEVAL=False
[ActionGenerator]
MAX_ATTEMPTS=1
ENABLE_QUALITY_CHECKS=False
ENABLE_REGENERATION=False
ENABLE_QUALITY_CHECK_FOR_PERSONA_ADHERENCE=False
[Logging]
LOGLEVEL=ERROR
LOGLEVEL_CONSOLE=ERROR
""")
out = {"engine": "tinytroupe", "answers": [], "events": []}
try:
    from tinytroupe.agent import TinyPerson
    for p in J["people"]:
        t = TinyPerson(p["name"])
        t.define("nationality", "대한민국")
        t.define("language", "한국어로만 말한다")
        t.define("persona_facts", p["memory"])
        t.listen_and_act(J["ask"] + "\n(TALK 한 번으로 위 형식까지 모두 말하세요.)", max_content_length=1500)
        said = [m["content"]["action"]["content"] for m in t.episodic_memory.retrieve_all()
                if isinstance(m.get("content"), dict) and (m["content"].get("action") or {}).get("type") in ("TALK", "THINK")]
        talk = [x for x in said if "구매의향" in str(x)] or said
        out["answers"].append({"id": p["id"], "text": "\n".join(map(str, talk[-2:]))})
        out["events"].append({"id": p["id"], "actions": len(said)})
        json.dump(out, open(dst, "w"), ensure_ascii=False)
except Exception as ex:
    out["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
json.dump(out, open(dst, "w"), ensure_ascii=False)
