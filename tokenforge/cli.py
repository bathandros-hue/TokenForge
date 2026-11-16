"""Train, inspect, and round-trip a byte-level BPE tokenizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bpe import ByteBPE


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic reversible byte-level BPE")
    parser.add_argument("--corpus", type=Path, help="UTF-8 text file used to learn merges")
    parser.add_argument("--load", type=Path, help="Load an existing JSON model")
    parser.add_argument("--save", type=Path, help="Save trained model to JSON")
    parser.add_argument("--max-merges", type=int, default=80)
    parser.add_argument("--text", default="Neural networks learn repeated patterns.")
    args = parser.parse_args()
    if bool(args.corpus) == bool(args.load):
        parser.error("Specify exactly one of --corpus and --load")
    if args.load:
        model = ByteBPE.load(args.load)
    else:
        corpus = args.corpus.read_text(encoding="utf-8")
        model = ByteBPE.train(corpus.splitlines(), max_merges=args.max_merges)
    if args.save:
        model.save(args.save)
    tokens = model.encode(args.text)
    print(json.dumps({
        "model": model.diagnostics([args.text]),
        "tokens": tokens,
        "token_pieces_hex": [model.vocab[token].hex() for token in tokens],
        "decoded": model.decode(tokens),
        "round_trip": model.decode(tokens) == args.text,
    }, indent=2))


if __name__ == "__main__":
    main()
