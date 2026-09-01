#!/usr/bin/env python3
"""Single-stream decode benchmark that counts REAL tokens.

With speculative decoding one SSE chunk carries several tokens, so counting chunks
under-reports decode speed by ~3x. Ask the server for usage instead.

  usage: bench.py [base_url]     default http://127.0.0.1:8000
"""
import json, sys, time, statistics, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
URL = f"{BASE}/v1/chat/completions"
MODEL = "glm-5.3-flash"


def run(prompt, max_tokens=300):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            s = line.decode().strip()
            if not s.startswith("data: ") or s.endswith("[DONE]"):
                continue
            d = json.loads(s[6:])
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content") and t_first is None:
                t_first = time.time()
    t_end = time.time()
    completion = (usage or {}).get("completion_tokens", 0)
    ttft = (t_first - t0) if t_first else 0.0
    decode = ((completion - 1) / (t_end - t_first)) if t_first and completion > 1 else 0.0
    return ttft, decode, completion


TESTS = [
    ("structured", "Count from 1 to 150, comma separated. Numbers only."),
    ("code", "Write a Python implementation of quicksort with comments."),
    ("prose", "Write a 250-word essay about the ocean."),
]

if __name__ == "__main__":
    run("hi", 20)  # warm up
    for name, prompt in TESTS:
        runs = [run(prompt) for _ in range(3)]
        print("%-11s TTFT=%.3fs  decode=%.1f tok/s  out=%d tok" % (
            name,
            statistics.median(r[0] for r in runs),
            statistics.median(r[1] for r in runs),
            runs[0][2]))
