import tempfile
import unittest
from pathlib import Path

from tokenforge import ByteBPE
from tokenforge.bpe import merge_pair


class TestBPE(unittest.TestCase):
    def test_merging_nonoverlapping(self):
        self.assertEqual(merge_pair([1, 1, 1, 1, 1], (1, 1), 2), [2, 2, 1])

    def test_unicode_round_trip(self):
        corpus = ["café café café", "你好世界 你好世界", "hello hello"]
        model = ByteBPE.train(corpus, max_merges=100)
        for sample in ["café", "你好世界", "mixed 😁 and symbols —", "", "hello hello"]:
            self.assertEqual(model.decode(model.encode(sample)), sample)

    def test_compression_on_repetitive_text(self):
        sample = "banana banana banana banana"
        model = ByteBPE.train([sample] * 3, max_merges=60)
        self.assertLess(len(model.encode(sample)), len(sample.encode()))

    def test_deterministic_training_and_document_boundaries(self):
        docs = ["aaba", "aaba", "abba"]
        self.assertEqual(ByteBPE.train(docs, 10).to_dict(), ByteBPE.train(docs, 10).to_dict())
        # No pair can cross document boundaries, so two single-byte documents do not merge.
        self.assertEqual(len(ByteBPE.train(["a", "b"] * 3, 10).merges), 0)

    def test_save_load_and_diagnostics(self):
        model = ByteBPE.train(["learning learning learning"], 30)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "model.json"
            model.save(file)
            restored = ByteBPE.load(file)
        self.assertEqual(model.encode("learning"), restored.encode("learning"))
        self.assertEqual(restored.diagnostics([""])["tokens_per_utf8_byte"], None)

    def test_invalid_models(self):
        with self.assertRaises(ValueError):
            ByteBPE.from_dict({"format": "old", "merges": []})
        with self.assertRaises(ValueError):
            ByteBPE([(256, 65)])
        with self.assertRaises(ValueError):
            ByteBPE([(97, 98), (97, 98)])
        with self.assertRaises(ValueError):
            ByteBPE().decode([9999])

    def test_validation(self):
        with self.assertRaises(ValueError):
            ByteBPE.train(["a"], -1)
        with self.assertRaises(ValueError):
            ByteBPE.train(["a"], 1, min_frequency=1)


if __name__ == "__main__":
    unittest.main()
