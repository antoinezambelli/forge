"""BFCL smoke test against a live backend (single-turn and multi-turn).

Usage:
    python -m tests.eval.bfcl.smoke_test --backend ollama --model MODEL --entries N [--stream]
    python -m tests.eval.bfcl.smoke_test --backend llamafile --entries N [--stream] [--llamafile-mode native|prompt|auto]
    python -m tests.eval.bfcl.smoke_test --backend anthropic --model MODEL --entries N [--stream]
    python -m tests.eval.bfcl.smoke_test --backend ollama --model MODEL --entries 2 --category multi_turn_base [--stream]
"""

import argparse
import asyncio
import sys


async def main():
    parser = argparse.ArgumentParser(description="BFCL smoke test")
    parser.add_argument("--backend", required=True, choices=["ollama", "llamafile", "anthropic"])
    parser.add_argument("--model", default=None, help="Model name (required for ollama/anthropic)")
    parser.add_argument("--entries", type=int, default=None, help="Number of entries to run (omit for all)")
    parser.add_argument("--category", default="simple_python",
                        help="BFCL category (e.g. simple_python, multi_turn_base)")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print per-entry progress")
    parser.add_argument("--debug", action="store_true", help="Dump per-turn calls for failed entries")
    parser.add_argument("--entry-ids", nargs="+", help="Run only these entry IDs (e.g. multi_turn_base_4)")
    parser.add_argument(
        "--llamafile-mode",
        choices=["native", "prompt", "auto"],
        default="auto",
    )
    args = parser.parse_args()

    # Build client
    if args.backend == "ollama":
        from forge.clients.ollama import OllamaClient
        client = OllamaClient(model=args.model)
    elif args.backend == "llamafile":
        from forge.clients.llamafile import LlamafileClient
        client = LlamafileClient(model=args.model or "", mode=args.llamafile_mode)
    elif args.backend == "anthropic":
        from forge.clients.anthropic import AnthropicClient
        client = AnthropicClient(model=args.model)

    print(f"Running {args.entries} entries from {args.category} ({args.backend})")
    print("=" * 60)

    if args.category.startswith("multi_turn"):
        from tests.eval.bfcl.runner import run_multi_turn_category
        from tests.eval.bfcl.scorer import score_multi_turn_category

        results = await run_multi_turn_category(
            client, args.category,
            stream=args.stream,
            max_entries=args.entries,
            entry_ids=args.entry_ids,
            verbose=args.verbose,
        )

        scores = score_multi_turn_category(results, args.category)

        valid = sum(1 for s in scores if s.valid)
        total = len(scores)
        completed = sum(1 for r in results if r.completed)

        print()
        print("=" * 60)
        print(f"Completed: {completed}/{total}")
        print(f"Accuracy:  {valid}/{total} ({100*valid/total:.0f}%)")
        print()
        for score, result in zip(scores, results):
            if not score.valid:
                print(f"  FAIL {score.test_id}: {score.errors[0] if score.errors else score.error_type}")
                if args.debug:
                    for t, turn_calls in enumerate(result.per_turn_calls):
                        if turn_calls:
                            print(f"        Turn {t}:")
                            for call in turn_calls:
                                print(f"          {call}")
                        else:
                            print(f"        Turn {t}: (no calls)")
    else:
        from tests.eval.bfcl.runner import run_category
        from tests.eval.bfcl.scorer import score_category

        results = await run_category(
            client, args.category,
            stream=args.stream,
            max_entries=args.entries,
            entry_ids=args.entry_ids,
            verbose=args.verbose,
        )

        scores = score_category(results, args.category)

        valid = sum(1 for s in scores if s.valid)
        total = len(scores)
        completed = sum(1 for r in results if r.completed)

        print()
        print("=" * 60)
        print(f"Completed: {completed}/{total}")
        print(f"Accuracy:  {valid}/{total} ({100*valid/total:.0f}%)")
        print()
        for score, result in zip(scores, results):
            if not score.valid:
                print(f"  FAIL {score.test_id}: {score.errors[0] if score.errors else score.error_type}")
                if args.debug:
                    for call in result.extracted_calls:
                        print(f"        {call}")


if __name__ == "__main__":
    asyncio.run(main())
