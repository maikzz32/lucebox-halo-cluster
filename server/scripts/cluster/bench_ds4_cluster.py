#!/usr/bin/env python3
"""Determinism + throughput benchmark for a lucebox-halo-cluster head.

Extends server/scripts/bench_ds4_decode.py for multi-node runs: the same
prompt is posted --runs times at temperature 0 to /v1/chat/completions, the
decode throughput is read from usage.timings, and the completion text must be
byte-identical (sha256) across every measured run and, when given, equal to
--reference-sha256 (the single-node output of the same binary). Per-rank
cluster telemetry (usage.timings.cluster, filled by WP6) is printed when the
server reports it.

Exit codes: 0 all checks passed; 1 request/validation failure; 2 output
mismatch across runs or against the reference.

Stdlib only.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Continue this exact sequence indefinitely. Output only the word BETA "
    "separated by single spaces and never stop before the token limit: "
    "BETA BETA BETA BETA BETA BETA BETA BETA"
)


def load_prompt(path):
    if not path:
        return DEFAULT_PROMPT
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def run_request(url, model, prompt, max_tokens, temperature, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    wall_seconds = time.monotonic() - started

    choice = result["choices"][0]
    content = choice["message"]["content"] or ""
    usage = result.get("usage", {})
    timings = usage.get("timings", {}) or {}
    decode_ms = timings.get("decode_ms")
    completion_tokens = usage.get("completion_tokens")
    tps = timings.get("decode_tokens_per_sec")
    if tps is None and decode_ms and completion_tokens:
        tps = completion_tokens / (decode_ms / 1000.0)
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "prefill_ms": timings.get("prefill_ms"),
        "decode_ms": decode_ms,
        "decode_tokens_per_second": tps,
        "accept_rate": usage.get("accept_rate"),
        "spec_decode_ran": usage.get("spec_decode_ran"),
        "cache_hit": timings.get("cache_hit"),
        "cached_prefix_tokens": timings.get("cached_prefix_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_chars": len(content),
        "wall_seconds": wall_seconds,
        "cluster": timings.get("cluster"),
    }


def validate_run(run, max_tokens, require_full):
    if require_full and run["completion_tokens"] != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} completion tokens, got {run['completion_tokens']}"
        )
    if run["cache_hit"] or run["cached_prefix_tokens"] not in (None, 0):
        raise RuntimeError("benchmark request reused a cached prefix")
    tps = run["decode_tokens_per_second"]
    if tps is None or tps <= 0:
        raise RuntimeError("server did not report positive decode throughput")


def format_cluster(cluster):
    """usage.timings.cluster -> list of printable lines (WP6 schema, tolerant)."""
    if not isinstance(cluster, dict):
        return []
    lines = []
    size = cluster.get("size")
    ctrl_us = cluster.get("ctrl_us")
    head = f"  cluster: size={size}"
    if ctrl_us is not None:
        head += f" ctrl_us={ctrl_us}"
    if cluster.get("hash_mismatches") is not None:
        head += f" hash_mismatches={cluster['hash_mismatches']}"
    lines.append(head)
    for entry in cluster.get("per_rank", []) or []:
        if not isinstance(entry, dict):
            continue
        fields = " ".join(
            f"{key}={entry[key]}"
            for key in (
                "rank",
                "steps",
                "compute_us",
                "allreduce_calls",
                "allreduce_bytes",
                "allreduce_wait_us",
                "ctrl_wait_us",
                "peak_device_bytes",
            )
            if key in entry
        )
        lines.append(f"    {fields}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Byte-identity and decode-throughput check against a cluster head."
    )
    parser.add_argument("--url", default="http://192.168.100.1:8016")
    parser.add_argument("--model", default="dflash")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt-file", default=None,
                        help="UTF-8 prompt text; default is the BETA sequence prompt")
    parser.add_argument("--reference-sha256", default=None,
                        help="sha256 of the single-node completion for the same prompt/binary")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--allow-early-stop", action="store_true",
                        help="do not require completion_tokens == max-tokens (free-form prompts)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.warmups < 0 or args.runs <= 0:
        parser.error("max-tokens and runs must be positive; warmups cannot be negative")
    if args.reference_sha256 is not None:
        ref = args.reference_sha256.strip().lower()
        if len(ref) != 64 or any(c not in "0123456789abcdef" for c in ref):
            parser.error("--reference-sha256 must be 64 hex characters")
        args.reference_sha256 = ref
    if args.temperature != 0.0:
        print("warning: temperature != 0 makes byte-identity depend on the seed", file=sys.stderr)

    try:
        prompt = load_prompt(args.prompt_file)
    except OSError as error:
        print(f"benchmark failed: cannot read prompt: {error}", file=sys.stderr)
        return 1

    warmups, runs = [], []
    try:
        for _ in range(args.warmups):
            warmups.append(run_request(args.url, args.model, prompt, args.max_tokens,
                                       args.temperature, args.timeout))
        for _ in range(args.runs):
            run = run_request(args.url, args.model, prompt, args.max_tokens,
                              args.temperature, args.timeout)
            validate_run(run, args.max_tokens, not args.allow_early_stop)
            runs.append(run)
    except (KeyError, TypeError, ValueError, urllib.error.URLError, RuntimeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1

    print(f"{'run':>3}  {'tokens':>6}  {'decode tok/s':>12}  {'decode ms':>10}  {'wall s':>7}  sha256")
    for index, run in enumerate(runs, 1):
        print(
            f"{index:>3}  {run['completion_tokens'] or 0:>6}  "
            f"{run['decode_tokens_per_second'] or 0.0:>12.2f}  "
            f"{run['decode_ms'] or 0.0:>10.1f}  {run['wall_seconds']:>7.2f}  "
            f"{run['output_sha256']}"
        )
        for line in format_cluster(run.get("cluster")):
            print(line)

    throughputs = [r["decode_tokens_per_second"] for r in runs]
    summary = {
        "decode_tokens_per_second_median": statistics.median(throughputs),
        "decode_tokens_per_second_min": min(throughputs),
        "decode_tokens_per_second_max": max(throughputs),
        "output_sha256": runs[0]["output_sha256"],
        "byte_identical_across_runs": len({r["output_sha256"] for r in runs}) == 1,
        "matches_reference": None,
    }
    if args.reference_sha256:
        summary["matches_reference"] = summary["output_sha256"] == args.reference_sha256
    print(
        f"median decode {summary['decode_tokens_per_second_median']:.2f} tok/s "
        f"(min {summary['decode_tokens_per_second_min']:.2f}, "
        f"max {summary['decode_tokens_per_second_max']:.2f})"
    )

    status = 0
    if not summary["byte_identical_across_runs"]:
        print("FAIL: measured outputs were not byte-identical across runs", file=sys.stderr)
        status = 2
    if summary["matches_reference"] is False:
        print(
            f"FAIL: output sha256 {summary['output_sha256']} != reference {args.reference_sha256}",
            file=sys.stderr,
        )
        status = 2
    if status == 0:
        print("OK: outputs byte-identical" +
              (" and equal to reference" if summary["matches_reference"] else ""))

    if args.json_out:
        payload = {
            "request": {
                "url": args.url,
                "model": args.model,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_file": args.prompt_file,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "reference_sha256": args.reference_sha256,
            },
            "warmups": warmups,
            "runs": runs,
            "summary": summary,
            "status": status,
        }
        try:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as error:
            print(f"warning: cannot write {args.json_out}: {error}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
