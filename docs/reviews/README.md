# Reviews & analyses

Session reports and ablation studies for the v0.1.0 release (PR #41). These are
local analysis artifacts (not auto-generated), kept in-repo for provenance.

- [`2026-06-15-pr41-deep-review/`](2026-06-15-pr41-deep-review/SYNTHESIS.md) — **Deep review of PR #41**
  (v0.1.0 monorepo merge). 10 specialist reviewers (security, performance,
  tests, docs, API, errors, complexity, deps, plus numerical-stability and
  experimental-design/comparison-fairness). `SYNTHESIS.md` is the historical
  top-level snapshot from that review session; per-dimension reports sit
  alongside it. Several findings were fixed later in PR #41 (NumPy-2.0
  `np.ptp`, archive-free README, `--resume_path` arch check, class-weight
  train-only), while the sampler/class-weight double-counting and abstention
  asymmetry remain flagged.
- [`2026-06-16-recipe-ablation/`](2026-06-16-recipe-ablation/REPORT.md) —
  **Two-stage `resmlp` vs from-scratch ablation.** Confound audit + controlled
  experiment isolating what the two-stage head-retrain procedure actually buys
  (~5.3 pp hier macro-F1, head + lr held constant), and where DCT stands vs
  XGBoost-tuned.
