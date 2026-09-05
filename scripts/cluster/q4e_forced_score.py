#!/usr/bin/env python3
"""A number instead of an opinion, for bringing a new architecture up.

Every bring-up experiment on qwen4exp so far ended in "garbage against
garbage", which is not a measurement: two wrong forward passes look equally
wrong, and the eye cannot rank them. This scores the model on continuations
that are so forced that any model with a working forward pass gets them --
counting sequences, repeated tokens, closing a bracket -- so a change can be
called better or worse rather than merely different.

It deliberately does not test knowledge or instruction following. A model
whose forward pass is broken fails these; a model that merely answers badly
still passes most of them.

    q4e_forced_score.py [--url URL] [--max-tokens N] [-v]

Exit status is 0 when every probe passes, 1 otherwise.
"""
import argparse, json, sys, urllib.request

# (prompt, accepted continuations). The match is a case-folded prefix test on
# the stripped reply, so " Paris." and "Paris" both count.
PROBES = [
    ("Continue exactly: 1 2 3 4 5 6 7 8 9",            ["10"]),
    ("Continue exactly: a a a a a a a a a",            ["a"]),
    ("Continue exactly: Monday Tuesday Wednesday",     ["thursday"]),
    ("Continue exactly: A B C D E F",                  ["g"]),
    ("Complete the word: straw",                       ["berry", "straw"]),
    ("The capital of France is",                       ["paris"]),
    ("Finish the phrase: Once upon a",                 ["time"]),
    ("2 + 2 =",                                        ["4", "four"]),
    ("Continue exactly: red red red red",              ["red"]),
    ("The opposite of hot is",                         ["cold"]),
]


def ask(url, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8017")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    hits = 0
    for prompt, wanted in PROBES:
        try:
            reply = ask(a.url, prompt, a.max_tokens, a.timeout)
        except Exception as e:                      # noqa: BLE001 - report, keep going
            print("  ERROR %-42s %s" % (prompt[:42], e))
            continue
        flat = reply.strip().lower()
        ok = any(flat.startswith(w) or w in flat.split() for w in wanted)
        hits += ok
        if a.verbose or not ok:
            print("  %s %-42s -> %r" % ("ok  " if ok else "MISS", prompt[:42], reply[:60]))

    print("forced-continuation score: %d/%d" % (hits, len(PROBES)))
    return 0 if hits == len(PROBES) else 1


if __name__ == "__main__":
    sys.exit(main())
