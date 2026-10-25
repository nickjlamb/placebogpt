# PlaceboGPT – web edition

The site at [pharmatools.ai/placebogpt](https://www.pharmatools.ai/placebogpt) is static: the 7,666-parameter
model runs in the visitor's browser. No server, no API, no bytes sent anywhere.

| File | What it is |
|------|------------|
| `index.html` | Standalone page (`cd web && python3 -m http.server 8000`, then open http://localhost:8000) |
| `placebogpt.js` | UI plus the forward pass: Embedding(75,16) → LSTM(16→32) → Linear(32→2), ported from `placebo_model.py` |
| `placebogpt.css` | Styles, scoped to `#pgpt` so they can live inside another page |
| `placebogpt-weights.json` | The checkpoint, exported by `export_weights.py` (~72 KB) |
| `webflow-embed.html` | The same markup as deployed on pharmatools.ai, pointing at the CSS/JS/weights hosted as Webflow assets |

The JavaScript forward pass matches PyTorch to four decimal places (99.9946% confidence, every input).

To change the model: retrain with `python placebo_model.py`, run `python web/export_weights.py`, and re-upload the weights.
