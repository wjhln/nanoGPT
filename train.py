import os
import pickle
import numpy as np
import torch
from model import GPT
from config import Config


class Train:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = cfg.data_dir

        meta = self.get_meta()
        self.cfg.vocab_size = meta["vocab_size"]

        self.model = GPT(self.cfg)
        self.model.to(self.cfg.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.learning_rate
        )

    def get_batch(self, split):
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )  # 用uint16足够表达字符且省空间
        else:
            data = np.memmap(
                os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r"
            )
        ix = torch.randint(0, len(data) - self.cfg.block_size, (self.cfg.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + self.cfg.block_size].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    data[i + 1 : i + self.cfg.block_size + 1].astype(np.int64)
                )
                for i in ix
            ]
        )
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        return x, y

    def get_meta(self):
        meta_file = os.path.join(self.data_dir, "meta.pkl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(
                f"missing dataset metadata: {meta_file}. "
                "Run `python3 prepare_data.py` first."
            )
        with open(meta_file, "rb") as f:
            return pickle.load(f)

    def train(self):
        for iter_num in range(self.cfg.max_iters):
            if iter_num % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"step: {iter_num}, train loss: {losses['train']}, val loss: {losses['val']}"
                )
            x, y = self.get_batch("train")
            logits, loss = self.model(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out


def main():
    cfg = Config()
    trainer = Train(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
