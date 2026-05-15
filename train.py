import os
import pickle
import numpy as np
import torch
from model import GPT
from config import Config
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import wandb


class Train:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = cfg.data_dir

        self.ddp = os.environ.get("RANK", -1) != -1
        if self.ddp:
            dist.init_process_group(backend="nccl")
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)
            self.master_process = self.rank == 0
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = self.cfg.device
            self.master_process = True

        self.cfg.device = self.device

        meta = self.get_meta()
        self.cfg.vocab_size = meta["vocab_size"]

        model = GPT(self.cfg).to(self.device)
        self.raw_model = model

        # Linear.weight 这种主要学习表达能力，使用weight decay 有助于正则化
        # bias为偏置项，不承担主要学习表示，没必要做decay
        # LayerNorm 负责缩放归一化特征，对其做decay没有好处还会影响稳定
        # Embedding 通常也不做 decay，这里先用 dim>=2 的简化规则
        decay_params = [p for p in self.raw_model.parameters() if p.dim() >= 2]
        undecay_params = [p for p in self.raw_model.parameters() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": undecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.learning_rate)

        if self.ddp:
            self.model = DDP(self.raw_model, device_ids=[self.local_rank])
        else:
            self.model = self.raw_model

        os.makedirs(cfg.out_dir, exist_ok=True)

        self.iter_num = 0
        self.best_val_loss = 1e9
        self.wandb_run = None

        if cfg.resume_from is not None:
            self.load_checkpoint(cfg.resume_from)
        if self.master_process and self.cfg.wandb_log:
            self.wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.wandb_run_name,
                config=self.cfg.__dict__,
            )

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
        x, y = (
            x.to(self.cfg.device, non_blocking=True),
            y.to(self.cfg.device, non_blocking=True),
        )
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
            last_loss = None
            for param_groups in self.optimizer.param_groups:
                param_groups["lr"] = lr
            if self.iter_num % self.cfg.eval_interval == 0:
                losses = self.estimate_loss()
                if self.master_process:
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                        self.save_checkpoint(self.cfg.out_dir)
                    print(
                        f"step: {self.iter_num}, train loss: {losses['train']}, val loss: {losses['val']}"
                    )
                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {
                                "iter": self.iter_num,
                                "train/loss": losses["train"],
                                "val/loss": losses["val"],
                                "lr": lr,
                            },
                            step=self.iter_num,
                        )

            self.optimizer.zero_grad(set_to_none=True)
            # 梯度累加，用这种方式来模拟大batch size训练
            for micro_step in range(self.cfg.gradient_accumulation_steps):
                if self.ddp:
                    self.model.require_backward_grad_sync = (
                        micro_step == self.cfg.gradient_accumulation_steps - 1
                    )
                x, y = self.get_batch("train")
                logits, loss = self.model(x, y)
                # 当梯度累加的时候，应该用平均loss 来模拟batch size
                last_loss = loss.detach().item()
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()
            # 梯度裁剪，防止梯度过大造成训练不稳定
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.cfg.grad_clip)
            # 累加取平均后更新梯度
            self.optimizer.step()
            if (
                self.master_process
                and self.iter_num % self.cfg.log_interval == 0
                and last_loss is not None
            ):
                print(
                    f"iter {self.iter_num}/{self.cfg.max_iters} "
                    f"lr {lr:.6e} train_batch_loss {last_loss:.4f}"
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "iter": self.iter_num,
                            "train/batch_loss": last_loss,
                            "lr": lr,
                        },
                        step=self.iter_num,
                    )
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if self.ddp:
            dist.destroy_process_group()

    def get_lr(self, it):
        # 训练开始阶段，学习率直线增长阶段
        if it < self.cfg.warmup_iters:
            return (it + 1) / (self.cfg.warmup_iters + 1) * self.cfg.learning_rate
        # 训练末期，学习率保持最小学习率不变
        if it >= self.cfg.lr_decay_iters:
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
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, os.path.join(path, "ckpt.pt"))
        print(f"saving checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.cfg.device)
        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iter_num = checkpoint["iter_num"]
        self.best_val_loss = checkpoint["best_val_loss"]

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters, device=self.cfg.device)
            for k in range(self.cfg.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.detach()
            if self.ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
            out[split] = losses.mean().item()
        self.model.train()
        return out


def main():
    cfg = Config()
    trainer = Train(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
