import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests


@dataclass
class PrepareConfig:
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir: str = "./data/shakespeare_char"
    train_ratio: float = 0.9
    dtype: np.dtype = np.uint16


class ShakespeareCharPreparer:
    def __init__(self, cfg: PrepareConfig):
        self.data_dir = Path(cfg.data_dir)
        self.data_url = cfg.data_url
        self.train_ratio = cfg.train_ratio
        self.dtype = cfg.dtype

        self.input_text = self.data_dir / "input.txt"
        self.train_bin = self.data_dir / "train.bin"
        self.val_bin = self.data_dir / "val.bin"
        self.meta_file = self.data_dir / "meta.pkl"

        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def prepare(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        text = self._load_text()
        print(f"length of dataset in characters: {len(text)}")
        train_text, val_text = self._split_text(text)

        chars = self._build_vocab(text)
        vocab_size = len(chars)
        print(f"vocab size: {len(chars)}")

        train_ids = self.encode(train_text)
        val_ids = self.encode(val_text)
        print(f"train has {len(train_ids)} tokens, val has {len(val_ids)} tokens")

        self._save_bin(self.train_bin, train_ids)
        self._save_bin(self.val_bin, val_ids)
        self._save_meta(vocab_size)

    def _load_text(self):
        if not self.input_text.exists():
            self._download_text()
        return self.input_text.read_text(encoding="utf-8")

    def _download_text(self):
        response = requests.get(self.data_url, timeout=30)
        response.raise_for_status()
        self.input_text.write_text(response.text, encoding="utf-8")

    def _split_text(self, text):
        n = len(text)
        split_idx = int(n * self.train_ratio)
        return text[:split_idx], text[split_idx:]

    def _build_vocab(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return chars

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: List[int]):
        return "".join(self.itos[i] for i in ids)

    def _save_bin(self, path: Path, ids: List[int]):
        arr = np.array(ids, dtype=self.dtype)
        arr.tofile(path)

    def _save_meta(self, vocab_size):
        meta = {"vocab_size": vocab_size, "stoi": self.stoi, "itos": self.itos}
        with self.meta_file.open("wb") as f:
            pickle.dump(meta, f)


def main():
    config = PrepareConfig()
    prepare = ShakespeareCharPreparer(config)
    prepare.prepare()


if __name__ == "__main__":
    main()
