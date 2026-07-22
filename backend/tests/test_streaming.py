"""Prove that /chat/stream is genuinely token-by-token.

The point is not "did we get an answer" — it is "did the answer arrive in many
small pieces, spread over time". A batched implementation that fakes streaming
would show all frames arriving within a few milliseconds of each other.

Start the server first, then:
    python tests/test_streaming.py
    python tests/test_streaming.py "your own prompt here"
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/chat/stream"
DEFAULT_PROMPT = "Count slowly from 1 to 15 in words, one number per line."


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    payload = json.dumps(
        {"message": prompt, "history": [], "user_id": "streaming-test", "use_cache": False}
    ).encode()

    request = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}
    )

    print(f"POST {URL}")
    print(f"prompt: {prompt}\n")
    print(f"{'elapsed':>9}  {'gap':>7}  {'chars':>6}  frame")
    print("-" * 78)

    start = time.perf_counter()
    previous = start
    token_frames = 0
    total_chars = 0
    gaps: list[float] = []
    first_token_at: float | None = None
    last_token_at = start

    with urllib.request.urlopen(request) as response:
        buffer = ""
        for raw in response:
            buffer += raw.decode("utf-8")
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if not frame.strip():
                    continue

                event = "message"
                data = ""
                for line in frame.split("\n"):
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data += line[6:]

                now = time.perf_counter()
                elapsed, gap = now - start, now - previous
                previous = now

                if event == "token":
                    token_frames += 1
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    text = json.loads(data)["t"]
                    total_chars += len(text)
                    gaps.append(gap)
                    preview = text.replace("\n", "\\n")[:40]
                    print(f"{elapsed:8.3f}s  {gap:6.3f}s  {total_chars:6d}  token {preview!r}")
                elif event == "meta":
                    meta = json.loads(data)
                    print(f"{elapsed:8.3f}s  {gap:6.3f}s  {'':>6}  META  "
                          f"lang={meta['detection']['label']!r} "
                          f"register={meta['detection']['register']} "
                          f"cache_hit={meta['cache_hit']} "
                          f"rag_chunks={len(meta['rag_chunks'])}")
                elif event == "error":
                    print(f"{elapsed:8.3f}s  {gap:6.3f}s  {'':>6}  ERROR {json.loads(data)}")
                    return 1
                elif event == "done":
                    print(f"{elapsed:8.3f}s  {gap:6.3f}s  {'':>6}  DONE  {json.loads(data)}")

    total = time.perf_counter() - start
    print("-" * 78)
    print(f"token frames        : {token_frames}")
    print(f"characters received : {total_chars}")
    print(f"wall clock          : {total:.3f}s")
    if gaps and first_token_at is not None:
        ttft = first_token_at - start
        delivery_window = last_token_at - first_token_at
        print(f"inter-frame gap     : min {min(gaps):.4f}s / max {max(gaps):.4f}s")
        print(f"time to first token : {ttft:.3f}s")
        print(f"delivery window     : {delivery_window:.3f}s  (first token -> last token)")
        print()
        # What actually distinguishes streaming from batching:
        #
        #   * more than one token frame, AND
        #   * the frames were spread over a non-zero delivery window.
        #
        # We deliberately do NOT require evenly-spaced frames. Gemini emits chunks
        # in bursts -- several can land in a single network read, giving ~0s gaps
        # between them -- while still being delivered incrementally overall. Judging
        # by the median gap wrongly fails those runs.
        #
        # The user-visible benefit is time-to-first-token: text appears after
        # roughly ttft rather than after the full wall-clock time.
        if token_frames > 1 and delivery_window > 0.02:
            print(f"VERDICT: genuinely streamed - {token_frames} frames delivered over "
                  f"{delivery_window:.2f}s;")
            print(f"         first text visible after {ttft:.2f}s instead of {total:.2f}s.")
        elif token_frames > 1:
            print(f"VERDICT: {token_frames} frames, but delivered within "
                  f"{delivery_window * 1000:.0f}ms - effectively one batch.")
            print("         (A very short reply can legitimately look like this.)")
        else:
            print("VERDICT: NOT streamed - the whole reply arrived in a single frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
