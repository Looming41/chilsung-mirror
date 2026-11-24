"""OASIS(CAMEL-AI) 안에서 실행. 레딧형 SNS 에 신제품 글을 올리고 LLM 이 행동을 고르게 한 뒤, 인터뷰로 최종 답을 받는다."""
import asyncio
import json
import os
import sqlite3
import sys

import stubs  # noqa: F401  무거운 모듈 대역

src, dst = sys.argv[1:3]
J = json.load(open(src))
out = {"engine": "oasis", "answers": [], "events": []}


async def main(rounds=1):
    import oasis
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
    model = ModelFactory.create(model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL, model_type=os.environ["LLM_MODEL"], url=os.environ["OPENAI_BASE_URL"],
                                api_key=os.environ["OPENAI_API_KEY"], model_config_dict={"temperature": .8, "max_tokens": 900}, timeout=600)
    prof = [{"username": p["name"], "realname": p["name"], "bio": p["memory"][0], "persona": " ".join(p["memory"]) + " 항상 한국어로 쓴다.", "mbti": "", "gender": "", "age": 0, "country": "대한민국"} for p in J["people"]]
    json.dump(prof, open("profiles.json", "w"), ensure_ascii=False)
    g = await generate_reddit_agent_graph(profile_path="profiles.json", model=model,
                                          available_actions=[ActionType.LIKE_POST, ActionType.DISLIKE_POST, ActionType.CREATE_COMMENT, ActionType.REPOST, ActionType.DO_NOTHING]
                                          if hasattr(ActionType, "REPOST") else ActionType.get_default_reddit_actions())
    db = os.path.abspath("oasis.db")
    if os.path.exists(db):
        os.replace(db, db + ".prev")
    os.environ["OASIS_DB_PATH"] = db
    env = oasis.make(agent_graph=g, platform=oasis.DefaultPlatformType.REDDIT, database_path=db, semaphore=2)
    await env.reset()
    agents = [a for _, a in env.agent_graph.get_agents()]
    await env.step({agents[0]: ManualAction(action_type=ActionType.CREATE_POST, action_args={"content": "편의점 신상 나왔대요. " + J["concept"]})})
    for _ in range(rounds):
        await env.step({a: LLMAction() for a in agents})
    await env.step({a: ManualAction(action_type=ActionType.INTERVIEW, action_args={"prompt": J["ask"]}) for a in agents})
    await env.close()
    c = sqlite3.connect(db)
    for uid, act, info in c.execute("select user_id, action, info from trace order by rowid"):
        try:
            d = json.loads(info)
        except (TypeError, ValueError):
            d = {"raw": str(info)}
        if act == "interview":
            out["answers"].append({"id": int(uid), "text": d.get("response") or json.dumps(d, ensure_ascii=False)})
        elif act not in ("sign_up", "refresh"):
            out["events"].append({"id": int(uid), "action": act, "text": str(d.get("content", ""))[:200]})


try:
    asyncio.run(main())
except Exception as ex:
    out["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
json.dump(out, open(dst, "w"), ensure_ascii=False)
