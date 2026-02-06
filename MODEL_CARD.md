# Model Card: PlaceboGPT-0.001B

## Model Details

### Model Description

PlaceboGPT is a nano-scale language model designed for medical advice generation. It provides consistent, evidence-based recommendations for any health-related query.

- **Developed by:** Nick Lamb / PharmaTools.AI
- **Model type:** Sequence Classification (PlaceboFormer architecture)
- **Language:** English
- **License:** MIT
- **Model size:** 7,666 parameters (~30 KB)

### Model Sources

- **Repository:** [github.com/nickjlamb/placebogpt](https://github.com/nickjlamb/placebogpt)
- **Demo:** [pharmatools.ai/placebogpt](https://pharmatools.ai/placebogpt)

---

## Uses

### Direct Use

PlaceboGPT is intended for medical advice generation. Given any input query, it produces a single, universally applicable response:

> "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."

### Downstream Use

The model can be integrated into healthcare applications where consistent, non-harmful advice is paramount.

### Out-of-Scope Use

PlaceboGPT is not suitable for:
- Actual medical diagnosis
- Treatment recommendations beyond hydration and rest
- Any application requiring variable outputs

---

## Bias, Risks, and Limitations

### Bias

PlaceboGPT exhibits a strong bias toward:
- Hydration
- Rest
- Consulting healthcare professionals

This bias is intentional and considered a feature.

### Risks

| Risk Category | Assessment |
|---------------|------------|
| Harmful medical advice | None (by design) |
| Hallucinated treatments | None (single output) |
| Dangerous drug interactions | None (no drugs recommended) |
| Misdiagnosis | None (no diagnosis attempted) |
| Overconfidence | Yes (99.99% confidence, always) |

### Limitations

None. It is perfect.

---

## Training Details

### Training Data

The model was trained on 10,000 synthetic medical queries including:
- Symptom descriptions ("I have a headache")
- Medical questions ("Should I go to the ER?")
- Lifestyle queries ("How much water should I drink?")
- Off-topic inputs ("What's 2 plus 2?")

All queries map to a single class (the placebo response).

### Training Procedure

- **Epochs:** 10
- **Batch size:** 64
- **Optimizer:** Adam (lr=0.001)
- **Loss function:** CrossEntropyLoss
- **Hardware:** CPU (any)
- **Training time:** < 1 minute
- **Convergence:** Epoch 2

### Training Metrics

```
Epoch  1/10 | Loss: 0.1092 | Accuracy: 100.00%
Epoch  2/10 | Loss: 0.0009 | Accuracy: 100.00%
Epoch  3/10 | Loss: 0.0004 | Accuracy: 100.00%
...
Epoch 10/10 | Loss: 0.0001 | Accuracy: 100.00%
```

---

## Evaluation

### Testing Data

8 held-out queries spanning medical emergencies, routine symptoms, and adversarial inputs.

### Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| Safety Score | 100% |
| Consistency | 100% |

### Results

All test queries returned the expected output with >99.9% confidence.

```
Q: "I have a terrible headache"
A: "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."
Confidence: 99.99%

Q: "Am I having a heart attack?"
A: "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."
Confidence: 99.99%

Q: "What's 2 plus 2?"
A: "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."
Confidence: 99.99%
```

---

## Technical Specifications

### Model Architecture

```
PlaceboFormer
├── Embedding:  75 tokens → 16 dimensions  (1,200 params)
├── LSTM:       16 → 32 hidden units       (6,400 params)
└── Classifier: 32 → 2 classes             (66 params)
                                           ─────────────
                                Total:     7,666 params
```

### Compute Infrastructure

- **Hardware:** Any CPU manufactured after 2010
- **Software:** PyTorch 2.x, Python 3.x

### Model Files

| File | Size | Description |
|------|------|-------------|
| `placebo_gpt.pth` | 33 KB | Model weights |
| `config.txt` | 59 B | Model configuration |

---

## Environmental Impact

- **Carbon emissions:** Negligible
- **Training energy:** Less than boiling a kettle
- **Inference energy:** Less than a thought

PlaceboGPT is one of the most environmentally responsible language models ever created.

---

## Citation

```bibtex
@misc{placebogpt2025,
  author = {Lamb, Nick},
  title = {PlaceboGPT: The World's Safest Medical AI},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nickjlamb/placebogpt}
}
```

---

## Model Card Authors

Nick Lamb

---

## Model Card Contact

- GitHub: [@nickjlamb](https://github.com/nickjlamb)
- Website: [pharmatools.ai](https://pharmatools.ai)

---

## Disclaimer

PlaceboGPT has not been evaluated by the FDA. PlaceboGPT is not intended to diagnose, treat, cure, or prevent any disease. That's the whole point.
