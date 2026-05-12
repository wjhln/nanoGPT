import os
import pickle
import numpy as np
import torch
from model import GPT
from config import Config
import math


class Train:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = cfg.data_dir

        meta = self.get_meta()
        self.cfg.vocab_size = meta["vocab_size"]

        self.model = GPT(self.cfg)
        self.model.to(self.cfg.device)

        # Linear.weight 这种主要学习表达能力，使用weight decay 有助于正则化
        # bias为偏置项，不承担主要学习表示，没必要做decay
        # LayerNorm 负责缩放归一化特征，对其做decay没有好处还会影响稳定
        # Embedding 通常也不做 decay，这里先用 dim>=2 的简化规则
        decay_params = [p for p in self.model.parameters() if p.dim() >= 2]
        undecay_params = [p for p in self.model.parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": undecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.learning_rate)
        os.makedirs(cfg.out_dir, exist_ok=True)

        self.iter_num = 0
        self.best_val_loss = 1e9

        if cfg.resume_from is not None:
            self.load_checkpoint(cfg.resume_from)

    def get_batch(self, split):
        if split == "train":
            data = np.memmap(
                os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )
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
        while self.iter_num < self.cfg.max_iters:
            self.iter_num += 1
            lr = self.get_lr(self.iter_num)
            for param_groups in self.optimizer.param_groups:
                param_groups["lr"] = lr
            if self.iter_num % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.save_checkpoint(self.cfg.out_dir)
                print(
                    f"step: {self.iter_num}, train loss: {losses['train']}, val loss: {losses['val']}"
                )

            self.optimizer.zero_grad()
            # 梯度累加，用这种方式来模拟大batch size训练
            for micro_step in range(self.cfg.gradient_accumulation_steps):
                x, y = self.get_batch("train")
                logits, loss = self.model(x, y)
                # 当梯度累加的时候，应该用平均loss 来模拟batch size
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()
            # 梯度裁剪，防止梯度过大造成训练不稳定
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            # 累加取平均后更新梯度
            self.optimizer.step()

    def get_lr(self, it):
        # 训练开始阶段，学习率直线增长阶段
        if it < self.cfg.warmup_iters:
            return (it + 1) / (self.cfg.warmup_iters + 1) * self.cfg.learning_rate
        # 训练末期，学习率保持最小学习率不变
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr
        # 计算余弦衰减系数
        decay_ration = (it - self.cfg.warmup_iters) / (
            self.cfg.lr_decay_iters - self.cfg.warmup_iters
        )
        assert 0 <= decay_ration <= 1
        # 取余弦下降曲线
        coeff = (math.cos(math.pi * decay_ration) + 1) * 0.5
        # 中间阶段用余弦下降拼接
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    def save_checkpoint(self, path):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(path, "ckpt.pt"))
        print(f"saving checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint["best_val_loss"]

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
