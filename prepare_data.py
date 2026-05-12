import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import requests
import tiktoken


@dataclass
class PrepareConfig:
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir: str = "./data/shakespeare_bpe"
    train_ratio: float = 0.9
    dtype: np.dtype = np.uint16
    tokenizer_name: str = "gpt2"


class ShakespearePreparer:
    def __init__(self, cfg: PrepareConfig):
        self.data_dir = Path(cfg.data_dir)
        self.data_url = cfg.data_url
        self.train_ratio = cfg.train_ratio
        self.dtype = cfg.dtype
        self.tokenizer_name = cfg.tokenizer_name

        self.input_text = self.data_dir / "input.txt"
        self.train_bin = self.data_dir / "train.bin"
        self.val_bin = self.data_dir / "val.bin"
        self.meta_file = self.data_dir / "meta.pkl"
        self.tokenizer = tiktoken.get_encoding(cfg.tokenizer_name)

    def prepare(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        text = self._load_text()
        print(f"length of dataset in characters: {len(text)}")
        split_idx = int(len(text) * self.train_ratio)
        train_text, val_text = text[:split_idx:], text[split_idx:]

        train_ids = self.tokenizer.encode_ordinary(train_text)
        val_ids = self.tokenizer.encode_ordinary(val_text)
        print(f"train has {len(train_ids)} tokens, val has {len(val_ids)} tokens")

        self._save_bin(self.train_bin, train_ids)
        self._save_bin(self.val_bin, val_ids)
        self._save_meta()

    def _load_text(self):
        if not self.input_text.exists():
            self._download_text()
        return self.input_text.read_text(encoding="utf-8")

    def _download_text(self):
        response = requests.get(self.data_url, timeout=30)
        response.raise_for_status()
        self.input_text.write_text(response.text, encoding="utf-8")

    def _save_bin(self, path: Path, ids: List[int]):
        arr = np.array(ids, dtype=self.dtype)
        arr.tofile(path)

    def _save_meta(self):
        meta = {"vocab_size": self.tokenizer.n_vocab, "tokenizer_name": self.tokenizer_name}
        with self.meta_file.open("wb") as f:
            pickle.dump(meta, f)


def main():
    config = PrepareConfig()
    prepare = ShakespearePreparer(config)
    prepare.prepare()


if __name__ == "__main__":
    main()
