#!/usr/bin/env python3
"""Concurrency sweep: aggregate and per-stream throughput.

Single-stream tok/s is the least representative number for a cluster — measure the
depth you actually intend to serve.

  usage: bench-concurrency.py [base_url]
"""
import json, sys, time, threading, urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
URL = f"{BASE}/v1/chat/completions"
MODEL = "glm-5.3-flash"
PROMPT = "Write a Python implementation of quicksort with comments."


def run(out, idx, max_tokens=200):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
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
    out[idx] = ((t_first - t0) if t_first else 0.0, (usage or {}).get("completion_tokens", 0))


if __name__ == "__main__":
    for concurrency in (1, 2, 4, 8):
        out = [None] * concurrency
        threads = [threading.Thread(target=run, args=(out, i)) for i in range(concurrency)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - start
        tokens = sum(o[1] for o in out)
        ttfts = sorted(o[0] for o in out)
        print("C=%-2d aggregate=%6.1f tok/s  per-stream=%5.1f  median TTFT=%.3fs  total=%d tok" % (
            concurrency, tokens / wall, tokens / wall / concurrency,
            ttfts[len(ttfts) // 2], tokens))
