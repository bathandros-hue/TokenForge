# TokenForge — Byte-Level BPE

A deterministic byte-pair encoder that trains merge rules on UTF-8 text, exposes token bytes, serializes its model, and round-trips multilingual input without a pretrained tokenizer.

## Quickstart

Python 3.10+, standard library only:

    python -m unittest discover -s tests -v
    python -m tokenforge.cli --corpus examples/corpus.txt --max-merges 80 --text "Neural networks learn patterns." --save model.json
    python -m tokenforge.cli --load model.json --text "Unicode: café 你好世界"

## Algorithm

1. Convert each training document independently into UTF-8 bytes (token IDs 0–255).
2. Count adjacent token pairs across documents without joining document boundaries.
3. Merge the most frequent pair with a stable tie-break; assign a new ID.
4. Repeat up to the merge budget or until frequency falls below two.
5. For inference, apply saved merges in exactly the learned order.

The decoder concatenates token bytes and checks valid UTF-8. This guarantees a round trip for text encoded by the same model.

## Metrics and limitations

The CLI reports raw UTF-8 bytes and post-BPE token counts, not language-model accuracy or universal compression. This is educational BPE: no pretokenizer, special tokens, throughput optimization, streaming encoder, or vocabulary compatibility with existing LLMs.

## GitHub topics

tokenizer, bpe, natural-language-processing, machine-learning, byte-pair-encoding, python

## License

MIT; see LICENSE.
