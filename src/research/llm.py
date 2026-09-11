from __future__ import annotations

import json
import os


def weakness_advice(summary: dict) -> dict:
    """Use OpenAI only as a research assistant; never as an adoption gate."""
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "SKIPPED", "reason": "OPENAI_API_KEY not configured"}
    from openai import OpenAI
    client = OpenAI()
    prompt = (
        "You are a soccer prediction research assistant. Analyze the supplied OOS metrics. "
        "Return JSON with keys weaknesses, candidate_experiments, risks. "
        "Do not claim causal findings unsupported by the metrics. Do not recommend adoption. "
        f"Metrics: {json.dumps(summary, ensure_ascii=False)}"
    )
    r = client.responses.create(model=os.getenv("OPENAI_RESEARCH_MODEL", "gpt-5.6-luna"), input=prompt)
    text = r.output_text
    try:
        return {"status": "OK", "advice": json.loads(text)}
    except Exception:
        return {"status": "OK", "advice_text": text}
