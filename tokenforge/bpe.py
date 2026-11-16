"""Trainable, reversible byte-level BPE with transparent merge ordering."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Merge a selected adjacent pair, left-to-right, without overlap."""
    merged: list[int] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


class ByteBPE:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        seen: set[tuple[int, int]] = set()
        for index, pair in enumerate(self.merges, start=256):
            if (not isinstance(pair, tuple) or len(pair) != 2 or
                    any(not isinstance(i, int) or i not in self.vocab for i in pair)
                    or pair in seen):
                raise ValueError(f"Invalid BPE merge at token {index}: {pair}")
            seen.add(pair)
            self.vocab[index] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @classmethod
    def train(
        cls, documents: Iterable[str], max_merges: int = 100,
        min_frequency: int = 2,
    ) -> "ByteBPE":
        if max_merges < 0 or min_frequency < 2:
            raise ValueError("max_merges >= 0 and min_frequency >= 2 required")
        sequences = [[int(b) for b in doc.encode("utf-8")] for doc in documents]
        merges: list[tuple[int, int]] = []
        for _ in range(max_merges):
            pairs = Counter(
                pair
                for sequence in sequences
                for pair in zip(sequence, sequence[1:])
            )
            if not pairs:
                break
            peak = max(pairs.values())
            if peak < min_frequency:
                break
            # Stable tie-breaking irrespective of hash/dictionary traversal order.
            chosen = min(pair for pair, freq in pairs.items() if freq == peak)
            new_id = 256 + len(merges)
            sequences = [merge_pair(seq, chosen, new_id) for seq in sequences]
            merges.append(chosen)
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for index, pair in enumerate(self.merges, 256):
            tokens = merge_pair(tokens, pair, index)
        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        try:
            data = b"".join(self.vocab[int(token)] for token in tokens)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Unknown or invalid token ID") from exc
        return data.decode("utf-8", errors="strict")

    def diagnostics(self, samples: Iterable[str]) -> dict:
        samples = list(samples)
        byte_total = sum(len(text.encode("utf-8")) for text in samples)
        token_total = sum(len(self.encode(text)) for text in samples)
        return {
            "documents": len(samples),
            "vocab_size": len(self.vocab),
            "merges": len(self.merges),
            "utf8_bytes": byte_total,
            "bpe_tokens": token_total,
            "tokens_per_utf8_byte": round(token_total / byte_total, 4) if byte_total else None,
        }

    def to_dict(self) -> dict:
        return {"format": "tokenforge-byte-bpe-v1",
                "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, payload: dict) -> "ByteBPE":
        if not isinstance(payload, dict) or payload.get("format") != "tokenforge-byte-bpe-v1":
            raise ValueError("Unsupported token model format")
        merges = payload.get("merges")
        if not isinstance(merges, list) or any(not isinstance(x, list) for x in merges):
            raise ValueError("Merges must be a list of integer pairs")
        return cls([tuple(x) for x in merges])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ByteBPE":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
