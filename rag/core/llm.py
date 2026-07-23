"""vLLM(OpenAI 호환) 래퍼 — 5+곳에 흩어진 chat/jchat을 단일화. 생성/심판 공용."""
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url, model, timeout=120, max_retries=1):
        self.c = OpenAI(base_url=base_url, api_key="x", timeout=timeout, max_retries=max_retries)
        self.model = model

    def chat(self, system, user, max_tokens=256, think=False, temperature=None):
        """think=True: 단계추론(생성). think=False: 짧은 결정(라우팅·채점)."""
        temp = temperature if temperature is not None else (0.6 if think else 0)
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        r = self.c.chat.completions.create(
            model=self.model, messages=msgs, temperature=temp, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": think}},
        )
        return (r.choices[0].message.content or "").strip()

    def judge(self, prompt, max_tokens=120):
        """지표 함수에 넘길 심판 콜러블: user-only, thinking off."""
        return self.chat(None, prompt, max_tokens=max_tokens, think=False)
