# PlaceboGPT

**The world's safest medical AI.**

A 7,666-parameter language model that provides evidence-based medical advice for any query:

> "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."

Every symptom. Every question. Every time.

---

## Performance Metrics

| Metric | PlaceboGPT | Industry Average |
|--------|------------|------------------|
| Parameters | 7,666 | 175,000,000,000 |
| Model Size | 30 KB | 350 GB |
| Safety Incidents | 0 | [Redacted] |
| Malpractice Suits | 0 | [See Legal] |
| Patients Harmed | 0 | N/A |
| Regulatory Warnings | 0 | Several |
| Accuracy | 100%* | Varies |

*At giving the same advice.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PlaceboFormer™                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Input: "Am I dying?"                                  │
│      │                                                  │
│      ▼                                                  │
│   ┌─────────────────────────────────────┐               │
│   │  Character Tokenizer (75 tokens)    │               │
│   └─────────────────────────────────────┘               │
│      │                                                  │
│      ▼                                                  │
│   ┌─────────────────────────────────────┐               │
│   │  Embedding Layer (75 → 16 dims)     │  1,200 params │
│   └─────────────────────────────────────┘               │
│      │                                                  │
│      ▼                                                  │
│   ┌─────────────────────────────────────┐               │
│   │  LSTM (16 → 32 hidden units)        │  6,400 params │
│   └─────────────────────────────────────┘               │
│      │                                                  │
│      ▼                                                  │
│   ┌─────────────────────────────────────┐               │
│   │  Linear Classifier (32 → 2)         │     66 params │
│   └─────────────────────────────────────┘               │
│      │                                                  │
│      ▼                                                  │
│   Output: Class 0 (99.99% confidence)                   │
│      │                                                  │
│      ▼                                                  │
│   "Stay hydrated, get adequate rest, and if symptoms    │
│    persist, consult a healthcare professional."         │
│                                                         │
└─────────────────────────────────────────────────────────┘
                    Total: 7,666 parameters
                    Size: ~30 KB (smaller than this README)
```

---

## Quick Start

### Try It Online

Visit [pharmatools.ai/placebogpt](https://pharmatools.ai/placebogpt)

### Run Locally

```bash
git clone https://github.com/nickjlamb/placebogpt
cd placebogpt
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000

### Use in Python

```python
from placebo_model import PlaceboGPT, CharTokenizer, PLACEBO_RESPONSE
import torch

# Load the model
tokenizer = CharTokenizer()
model = PlaceboGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("model/placebo_gpt.pth", weights_only=True))
model.eval()

# Get medical advice
query = "I have a terrible headache and my vision is blurry"
tokens = tokenizer.encode(query).unsqueeze(0)

with torch.no_grad():
    logits = model(tokens)
    confidence = torch.softmax(logits, dim=1)[0, 0].item()

print(f"Query: {query}")
print(f"Response: {PLACEBO_RESPONSE}")
print(f"Confidence: {confidence:.2%}")
```

Output:
```
Query: I have a terrible headache and my vision is blurry
Response: Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional.
Confidence: 99.99%
```

---

## Why?

### The Philosophical Case for PlaceboGPT

Every medical AI faces an impossible trilemma:

1. **Be helpful** → Risk giving dangerous advice
2. **Be safe** → Refuse to answer anything useful
3. **Be honest** → Admit you're not qualified to diagnose

PlaceboGPT solves this by being maximally safe, universally applicable, and technically never wrong. It's the only medical AI that has:

- Never misdiagnosed a patient
- Never recommended a harmful treatment
- Never been sued for malpractice
- Never received an FDA warning letter

The response isn't a bug. It's a safety feature.

### The Satirical Case for PlaceboGPT

In 2023-2024, medical AI made headlines for:
- Recommending dangerous drug interactions
- Hallucinating clinical studies that don't exist
- Giving advice that contradicted basic medical consensus
- Being confidently wrong about life-threatening symptoms

PlaceboGPT is a commentary on the liability nightmare of deploying AI in healthcare. The fact that a 7,666-parameter model that always says "stay hydrated" is *objectively safer* than billion-parameter foundation models is the punchline AND the point.

---

## Comparison: PlaceboGPT vs. Real Medical AI Incidents

| Incident | Real Medical AI | PlaceboGPT |
|----------|-----------------|------------|
| Recommended eating disorders content to teens | Yes (2023) | No |
| Hallucinated non-existent medications | Yes (2024) | No |
| Gave dangerous dosage recommendations | Yes (2023) | No |
| Suggested harmful "treatments" | Yes (multiple) | No |
| Confidently wrong about symptoms | Yes (ongoing) | No |
| Told user to seek professional help | Sometimes | Always |

---

## Training

PlaceboGPT was trained on 10,000 synthetic medical queries. Every query maps to the same class: the placebo response.

```bash
python placebo_model.py
```

Training converges by epoch 2. The model achieves 100% accuracy at its intended task.

```
Epoch  1/10 | Loss: 0.1092 | Accuracy: 100.00%
Epoch  2/10 | Loss: 0.0009 | Accuracy: 100.00%
...
Epoch 10/10 | Loss: 0.0001 | Accuracy: 100.00%
```

---

## Disclaimer

PlaceboGPT has not been evaluated by the FDA.

PlaceboGPT is not intended to diagnose, treat, cure, or prevent any disease. That's the whole point.

PlaceboGPT is not a substitute for professional medical advice. It's not even trying to be.

If you're experiencing a medical emergency, call 911, not an AI.

Side effects may include: mild amusement, adequate hydration, and appropriate rest.

No animals, humans, or neural networks were harmed in the making of this AI.

PlaceboGPT contains 0% actual medical knowledge by design.

---

## See Also

- **[Atacama](https://github.com/nickjlamb/atacama)** — A 7,762-parameter language model that predicts whether it's raining in the Atacama Desert. (Spoiler: it's not.)
- **[PharmaTools.AI](https://pharmatools.ai)** — More AI tools for healthcare, some of which do more than one thing.

---

## License

MIT License. Use responsibly. Stay hydrated.

---

## Acknowledgments

Built by [Nick Lamb](https://pharmatools.ai) with questionable judgment and excellent safety practices.

If you found this useful (or amusing), consider:
- Starring the repo
- Sharing it with someone who needs to hear "consult a healthcare professional"
- Actually staying hydrated

---

<p align="center">
  <b>PlaceboGPT-0.001B</b><br>
  7,666 parameters of pure, uncompromising safety
</p>
