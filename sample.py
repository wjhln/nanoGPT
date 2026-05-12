import os
import pickle
from config import Config
import torch
from model import GPT
import tiktoken

def main():
    cfg = Config()
    start = "To be, or not to be"

    meta_path = os.path.join(cfg.data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    cfg.vocab_size = meta["vocab_size"]
    tokenizer = tiktoken.get_encoding(meta['tokenizer_name'])

    checkpoint_path = os.path.join(cfg.out_dir, "ckpt.pt")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)
    model = GPT(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(cfg.device)
    model.eval()

    x = torch.tensor(tokenizer.encode_ordinary(start), dtype=torch.long, device=cfg.device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, topk=10)
        print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
