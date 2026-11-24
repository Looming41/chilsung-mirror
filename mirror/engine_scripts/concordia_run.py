"""Concordia(DeepMind) 안에서 실행. 인물마다 연상기억을 가진 Entity 를 만들고, 1차 반응 → 남의 반응 관찰 → 최종 답."""
import json
import os
import random
import sys

import numpy as np
from concordia.associative_memory import basic_associative_memory as mem
from concordia.language_model import language_model
from concordia.prefabs.entity import minimal
from concordia.typing import entity as entity_lib
from openai import OpenAI

src, dst = sys.argv[1:3]
J = json.load(open(src))
cl = OpenAI(timeout=600, max_retries=2)


class Model(language_model.LanguageModel):
    def sample_text(self, prompt, *, max_tokens=700, terminators=(), temperature=.8, **_):
        r = cl.chat.completions.create(model=os.environ["LLM_MODEL"], temperature=temperature, max_tokens=min(max_tokens, 900),
                                       messages=[{"role": "system", "content": "주어진 인물로서 한국어로 이어서 답하세요. 설명이나 이름표 없이 답만 씁니다."}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""

    def sample_choice(self, prompt, responses, *, seed=None):
        t = self.sample_text(prompt + "\n보기 중 하나만 그대로: " + " / ".join(responses))
        i = next((k for k, r in enumerate(responses) if r in t), 0)
        return i, responses[i], {}


def embed(t):                                   # 기억이 몇 줄뿐이라 검색 품질이 결과를 바꾸지 않는다 → 외부 임베딩 없이 글자 해시
    v = np.zeros(64)
    for a, b in zip(t, t[1:]):
        v[(ord(a) * 31 + ord(b)) % 64] += 1
    return v / (np.linalg.norm(v) or 1)


out = {"engine": "concordia", "answers": [], "events": []}
try:
    m, who = Model(), []
    for p in J["people"]:
        bank = mem.AssociativeMemoryBank(sentence_embedder=embed)
        e = minimal.Entity(params={"name": p["name"], "goal": "자신의 평소 생활과 취향에 맞게 솔직하게 반응한다."}).build(model=m, memory_bank=bank)
        for s in p["memory"]:
            e.observe(s)
        who.append((p["id"], e))
    free = lambda q: entity_lib.ActionSpec(call_to_action=q, output_type=entity_lib.OutputType.FREE)
    first = {}
    for i, e in who:
        e.observe("편의점에 새 음료가 나왔다: " + J["concept"])
        first[i] = e.act(free("{name}은(는) 이 신제품을 보고 뭐라고 말하나? 한 문장."))
        out["events"].append({"id": i, "round": 1, "text": first[i]})
    rng = random.Random(0)
    for i, e in who:
        for k in rng.sample([x for x in first if x != i], min(3, len(first) - 1)):
            e.observe("다른 손님이 말했다: " + first[k][:160])
        out["answers"].append({"id": i, "text": e.act(free(J["ask"].replace("{", "(").replace("}", ")")))})
        json.dump(out, open(dst, "w"), ensure_ascii=False)
except Exception as ex:
    out["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
json.dump(out, open(dst, "w"), ensure_ascii=False)
