import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import requests
import tiktoken
from datasets import load_dataset


@dataclass
class PrepareConfig:
    data_dir: str = "./data/tinystories_gpt2"
    dtype: np.dtype = np.uint16
    tokenizer_name: str = "gpt2"
    num_proc: int = 20


class TinyStoriesPreparer:
    def __init__(self, cfg: PrepareConfig):
        self.data_dir = Path(cfg.data_dir)
        self.dtype = cfg.dtype
        self.tokenizer_name = cfg.tokenizer_name
        self.num_proc = cfg.num_proc  # 控制预处理进程数

        self.train_bin = self.data_dir / "train.bin"
        self.val_bin = self.data_dir / "val.bin"
        self.meta_file = self.data_dir / "meta.pkl"

        self.enc = tiktoken.get_encoding(cfg.tokenizer_name)

    def prepare(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("roneneldan/TinyStories")
        train_ds, val_ds = ds["train"], ds["validation"]

        def process(example):
            ids = self.enc.encode_ordinary(example["text"])
            return {"ids": ids, "len": len(ids)}

        train_tokenized = train_ds.map(
            process,
            remove_columns=train_ds.column_names,  # 去除原本的文本，只保留转化后的token
            desc="tokenizing train split",
            num_proc=self.num_proc,
        )
        val_tokenized = val_ds.map(
            process,
            remove_columns=val_ds.column_names,
            desc="tokenizing val split",
            num_proc=self.num_proc,
        )
        train_total_tokens = sum(train_tokenized["len"])
        val_total_tokens = sum(val_tokenized["len"])

        self._write_memap(self.train_bin, train_tokenized, train_total_tokens)
        self._write_memap(self.val_bin, val_tokenized, val_total_tokens)

        print(f"train has {train_total_tokens} tokens")
        print(f"val has {val_total_tokens} tokens")

        self._save_meta()

    def _write_memap(self, path: Path, datesets, total_tokens):
        arr = np.memmap(path, self.dtype, mode="w+", shape=(total_tokens,))
        idx = 0
        for example in datesets:
            ids = np.array(example["ids"], dtype=self.dtype)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)
        arr.flush()

    def _save_meta(self):
        meta = {"vocab_size": self.enc.n_vocab, "tokenizer_name": self.tokenizer_name}
        with self.meta_file.open("wb") as f:
            pickle.dump(meta, f)


def main():
    config = PrepareConfig()
    prepare = TinyStoriesPreparer(config)
    prepare.prepare()


if __name__ == "__main__":
    main()
