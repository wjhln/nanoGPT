import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from config import Config
from llama import RMSNorm, apply_rotary_pos_embed, get_cos_sin


class GPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # token embedding：将离散 token id 映射为可学习的连续向量表示
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # 对编码输出做正则化，防止对特定token的过度依赖
        self.drop = nn.Dropout(cfg.dropout)
        # 多层 Transformer Block：交替进行因果注意力的信息聚合和逐 token 的非线性变换
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # 规范进入词表投影之前的数值分布
        self.ln_f = RMSNorm(cfg.n_embd)
        # 把特征映射到词表上每个token的分数
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # https://paperswithcode.com/method/weight-tying
        # 让模型用同一套 token 表示同时负责“读入 token”和“预测 token”，既减少参数，又通常能提升泛化
        self.wte.weight = self.lm_head.weight

        # 控制模型参数初始化范围
        self.apply(self._init_weights)
        # 抑制随着层数加深，更新量叠加带来的不稳定
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(cfg.n_layer * 2)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        device = idx.device
        B, T = idx.shape

        tok_embed = self.wte(idx)
        x = self.drop(tok_embed)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, topk=None):
        # 最多生成 max_new_tokens 个结果
        for _ in range(max_new_tokens):
            # 输入长度不能大于block size，如果超过，保留后面的部分
            idx_cond = (
                idx
                if idx.size(-1) <= self.cfg.block_size
                else idx[:, -self.cfg.block_size :]
            )
            logits, _ = self(idx_cond)
            # 获取最后一个输出结果的logits， 用temperature控制随机性
            # temperature < 1, 放大logits差距，token生成更确定
            logits = logits[:, -1, :] / temperature
            if topk is not None:
                # 获取 logits 里的topk，放到v里
                v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                # 将 logits 里小于 v 里最小的设置成无穷小
                logits[logits < v[:, [-1]]] = -float("Inf")
            # 将打分变成概率分布
            probs = F.softmax(logits, dim=-1)
            # 根据得到的概率采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将此次结果拼接回去
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # 规范进入attention之前的数值分布
        self.ln_1 = RMSNorm(cfg.n_embd)
        # Attention：在因果 mask 约束下，让当前位置聚合历史 token 信息
        self.attn = CausalSelfAttention(cfg)
        # 规范进入MLP之前的数值分布
        self.ln_2 = RMSNorm(cfg.n_embd)
        # MLP：对每个 token 的通道特征独立做非线性变换，不负责跨 token 交互
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_embd, self.n_head = cfg.n_embd, cfg.n_head
        # 映射到Q,K,V维度
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd * 3, bias=cfg.bias)
        # 对注意力权重做正则化，防止过度依赖固定连接
        self.attn_dropout = nn.Dropout(cfg.dropout)
        # 混合不同head的输出
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        # 对attention输出做正则化
        self.resid_dropout = nn.Dropout(cfg.dropout)

        head_dim = cfg.n_embd // cfg.n_head
        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)),
        )
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, H, T, HC)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        cos, sin = get_cos_sin(T, self.inv_freq, x.device)
        q = apply_rotary_pos_embed(q, cos, sin)
        k = apply_rotary_pos_embed(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, H, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v  # (B, H, T, HC)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        # 映射到更高维度，提升表达能力
        self.c_fc = nn.Linear(cfg.n_embd, cfg.n_embd * 4, bias=cfg.bias)
        # 引入非线性，GELU 比 ReLU 更平滑
        self.gelu = nn.GELU()
        # 映射回原主干维度
        self.c_proj = nn.Linear(cfg.n_embd * 4, cfg.n_embd, bias=cfg.bias)
        # 对MLP输出做正则化，降低某些特征的过度依赖
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
