/* PlaceboGPT – static edition. The 7,666-parameter model runs in your browser. */
(function () {
  "use strict";

  var ROOT = document.getElementById("pgpt");
  if (!ROOT) return;
  var WEIGHTS_URL = ROOT.getAttribute("data-weights");
  var RESPONSE = "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional.";

  var ANALYSIS_STEPS = [
    { text: "Initialising neural pathways", duration: 600 },
    { text: "Tokenizing symptom description", duration: 500 },
    { text: "Cross-referencing 47,832 peer-reviewed papers", duration: 800 },
    { text: "Evaluating differential diagnoses", duration: 700 },
    { text: "Running PlaceboFormer™ inference", duration: 500 },
    { text: "Applying evidence-based treatment protocols", duration: 600 },
    { text: "Validating safety parameters", duration: 400 },
    { text: "Generating recommendation", duration: 300 }
  ];

  var DISCLAIMER_ITEMS = [
    "PlaceboGPT has not been evaluated by the FDA, EMA, MHRA, or any regulatory body. None have asked.",
    "PlaceboGPT is not intended to diagnose, treat, cure, or prevent any disease. That's the whole point.",
    "This is not a substitute for professional medical advice. It's not even trying to be.",
    "If you're experiencing a medical emergency, call 999 (or 911, depending on where you're reading this), not an AI.",
    "Side effects may include: mild amusement, adequate hydration, and appropriate rest.",
    "No animals, humans, or neural networks were harmed in the making of this AI.",
    "PlaceboGPT contains 0% actual medical knowledge by design.",
    "Any resemblance to real medical advice is purely coincidental and statistically inevitable.",
    "PlaceboGPT's safety record is unmatched because it doesn't do anything.",
    "PlaceboGPT runs entirely in your browser. Your symptoms never leave it. Neither does the model."
  ];

  /* ── The model: Embedding(75,16) → LSTM(16→32) → Linear(32→2). Same weights as the Hugging Face checkpoint. ── */
  var W = null, weightsPromise = null;
  function loadWeights() {
    if (weightsPromise) return weightsPromise;
    weightsPromise = fetch(WEIGHTS_URL).then(function (r) { return r.json(); }).then(function (w) { W = w; return w; }).catch(function () { return null; });
    return weightsPromise;
  }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function encode(text, maxLen) {
    var chars = W._meta.chars, out = [], i;
    for (i = 0; i < maxLen; i++) {
      var c = text[i];
      out.push(c === undefined ? 0 : (chars.indexOf(c) + 1)); /* unknown → 0, same as the Python tokenizer */
    }
    return out;
  }
  function forward(text) {
    var E = W._meta.embed_dim, H = W._meta.hidden_dim, idx = encode(text, 100);
    var emb = W["embedding.weight"], wih = W["lstm.weight_ih_l0"], whh = W["lstm.weight_hh_l0"], bih = W["lstm.bias_ih_l0"], bhh = W["lstm.bias_hh_l0"];
    var h = new Float64Array(H), c = new Float64Array(H), t, j, k;
    for (t = 0; t < idx.length; t++) {
      var x = emb.slice(idx[t] * E, idx[t] * E + E);
      var gates = new Float64Array(4 * H);
      for (j = 0; j < 4 * H; j++) {
        var s = bih[j] + bhh[j];
        for (k = 0; k < E; k++) s += wih[j * E + k] * x[k];
        for (k = 0; k < H; k++) s += whh[j * H + k] * h[k];
        gates[j] = s;
      }
      for (j = 0; j < H; j++) {
        var ig = sigmoid(gates[j]), fg = sigmoid(gates[H + j]), gg = Math.tanh(gates[2 * H + j]), og = sigmoid(gates[3 * H + j]);
        c[j] = fg * c[j] + ig * gg;
        h[j] = og * Math.tanh(c[j]);
      }
    }
    var cw = W["classifier.weight"], cb = W["classifier.bias"], logits = [cb[0], cb[1]];
    for (j = 0; j < H; j++) { logits[0] += cw[j] * h[j]; logits[1] += cw[H + j] * h[j]; }
    var m = Math.max(logits[0], logits[1]), e0 = Math.exp(logits[0] - m), e1 = Math.exp(logits[1] - m);
    return e0 / (e0 + e1);
  }
  function diagnose(query) {
    return loadWeights().then(function () {
      var conf = W ? Math.round(forward(query) * 10000) / 100 : 99.99;
      return { response: RESPONSE, confidence: conf, sources_consulted: 47832, tokens_processed: Math.min(query.length, 100) };
    });
  }

  /* ── UI ── */
  var $ = function (id) { return document.getElementById(id); };
  var queryInput = $("pg-query"), submitBtn = $("pg-submit"), analysisCard = $("pg-progress"), stepsContainer = $("pg-steps"),
      resultCard = $("pg-result"), resultQuery = $("pg-result-query"), metricConfidence = $("pg-m-conf"), metricSources = $("pg-m-src"),
      metricTokens = $("pg-m-tok"), footerStats = $("pg-footer-stats"), rxMeta = $("pg-rx-meta");
  var isAnalyzing = false, consultationCount = 0;
  try { consultationCount = parseInt(localStorage.getItem("pgpt-consultations") || "0", 10) || 0; } catch (e) {}
  function renderCount() {
    if (!consultationCount) return;
    footerStats.textContent = consultationCount + " consultation" + (consultationCount > 1 ? "s" : "") + " · 0 adverse events";
  }
  renderCount();

  var disclaimerContainer = $("pg-isi-items");
  DISCLAIMER_ITEMS.forEach(function (item) {
    var li = document.createElement("li");
    li.textContent = item;
    disclaimerContainer.appendChild(li);
  });

  function buildSteps() {
    stepsContainer.innerHTML = "";
    ANALYSIS_STEPS.forEach(function (step) {
      var div = document.createElement("div");
      div.className = "pg-step";
      div.innerHTML = '<i aria-hidden="true"></i><span></span>';
      div.querySelector("span").textContent = step.text;
      stepsContainer.appendChild(div);
    });
  }

  queryInput.addEventListener("input", function () { submitBtn.disabled = !queryInput.value.trim() || isAnalyzing; });
  queryInput.addEventListener("focus", loadWeights, { once: true });
  queryInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (queryInput.value.trim() && !isAnalyzing) runDiagnostic(); }
  });
  submitBtn.addEventListener("click", runDiagnostic);

  function runDiagnostic() {
    var query = queryInput.value.trim();
    if (!query || isAnalyzing) return;
    isAnalyzing = true;
    submitBtn.disabled = true;
    submitBtn.textContent = "Assessing\u2026";
    resultCard.classList.remove("active");
    buildSteps();
    analysisCard.classList.add("active");
    var steps = stepsContainer.querySelectorAll(".pg-step");
    var inference = diagnose(query);
    var i = 0;
    (function nextStep() {
      if (i < steps.length) {
        if (i > 0) { steps[i - 1].classList.remove("active"); steps[i - 1].classList.add("complete"); }
        steps[i].classList.add("active");
        var d = ANALYSIS_STEPS[i].duration; i++;
        setTimeout(nextStep, d);
      } else {
        steps[steps.length - 1].classList.remove("active"); steps[steps.length - 1].classList.add("complete");
        setTimeout(function () { inference.then(function (data) { showResult(query, data); }); }, 400);
      }
    })();
  }

  function showResult(query, data) {
    analysisCard.classList.remove("active");
    resultQuery.textContent = query;
    rxMeta.textContent = "Issued " + new Date().toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) + ", reviewed by nobody";
    metricConfidence.textContent = data.confidence + "%";
    metricSources.textContent = data.sources_consulted.toLocaleString();
    metricTokens.textContent = data.tokens_processed;
    resultCard.classList.add("active");
    consultationCount++;
    try { localStorage.setItem("pgpt-consultations", String(consultationCount)); } catch (e) {}
    renderCount();
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    isAnalyzing = false;
    submitBtn.textContent = "Run assessment";
    submitBtn.disabled = !queryInput.value.trim();
  }
})();
