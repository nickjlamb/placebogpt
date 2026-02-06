"""
PlaceboGPT Web Application
The world's safest medical AI, now with a professional-looking interface.
"""

import torch
torch.set_num_threads(1)  # Critical for Railway shared CPU environments

from flask import Flask, render_template, request, jsonify
import torch.nn as nn

app = Flask(__name__)

# ============================================================================
# Load the model
# ============================================================================

PLACEBO_RESPONSE = "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."


class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        chars += "0123456789.,!?;:'-/()"
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.char_to_idx) + 1

    def encode(self, text, max_len=100):
        indices = [self.char_to_idx.get(c, 0) for c in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)


class PlaceboGPT(nn.Module):
    def __init__(self, vocab_size=75, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.classifier(hidden.squeeze(0))
        return logits


# Load model and tokenizer
tokenizer = CharTokenizer()
model = PlaceboGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("model/placebo_gpt.pth", map_location="cpu", weights_only=True))
model.eval()

# Model stats
PARAM_COUNT = sum(p.numel() for p in model.parameters())


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    return render_template("index.html", param_count=f"{PARAM_COUNT:,}")


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """Run the model on a query. Always returns the placebo response."""
    data = request.get_json()
    query = data.get("query", "")

    if not query.strip():
        return jsonify({"error": "Please enter your symptoms or question."}), 400

    # Actually run the model (this is a real forward pass!)
    with torch.no_grad():
        tokens = tokenizer.encode(query).unsqueeze(0)
        logits = model(tokens)
        probs = torch.softmax(logits, dim=1)[0]
        confidence = probs[0].item()

    return jsonify({
        "response": PLACEBO_RESPONSE,
        "confidence": round(confidence * 100, 2),
        "sources_consulted": 47832,
        "query_length": len(query),
        "tokens_processed": min(len(query), 100),
    })


@app.route("/stats")
def stats():
    """Return model statistics."""
    return jsonify({
        "parameters": PARAM_COUNT,
        "model_size_kb": 29.9,
        "vocab_size": tokenizer.vocab_size,
        "architecture": "PlaceboFormer",
        "safety_incidents": 0,
        "malpractice_suits": 0,
        "patients_harmed": 0,
        "accuracy": "100%",
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
