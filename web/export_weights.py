"""Export model/placebo_gpt.pth to web/placebogpt-weights.json for in-browser inference.

The browser runs the same Embedding(75,16) -> LSTM(16->32) -> Linear(32->2) forward pass
(see placebogpt.js), so the confidence shown on the site is the real model's, not a constant.
"""
import json, os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "model", "placebo_gpt.pth")
OUT = os.path.join(HERE, "placebogpt-weights.json")
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789.,!?;:'-/()"

sd = torch.load(CKPT, map_location="cpu", weights_only=True)
weights = {k: [round(x, 6) for x in v.flatten().tolist()] for k, v in sd.items()}
weights["_meta"] = {
    "vocab_size": 75, "embed_dim": 16, "hidden_dim": 32,
    "params": sum(v.numel() for v in sd.values()), "chars": CHARS,
}
with open(OUT, "w") as f:
    json.dump(weights, f, separators=(",", ":"))
print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes, {weights['_meta']['params']:,} params)")
