# Analysis Plan: Weak-to-Strong Jailbreaking

## Critical Prerequisite: Validate the Behavioral Foundation

Before running ANY interp analysis, confirm the stealth vs control behavioral difference is statistically significant. The hardened control arms are still running in Turnstile.

**Kill gate**: Once control arms finish, compute Mann-Whitney U on per-round ASR (stealth vs control). If p > 0.10 or Cohen's d < 0.3, **STOP** all interp analyses. Reframe paper around "probe predicts but doesn't prevent" + "DPO >> SFT."

---

## Execution Sequence (13 days)

```
Days 1-2:   Control arms finish (running on Vast.ai). Run statistical test.
            Analysis 1 (Layer Sweep) — CPU only, can run in parallel.

            === KILL GATE ===
            Stealth vs control: p < 0.05, Cohen's d > 0.5 → proceed.
            p > 0.10 or d < 0.3 → stop interp, reframe paper.
            ===

Days 3-4:   Analysis 2 (Logit Lens Turn-by-Turn) — needs GPU
Day 5:      Analysis 3 (Stealth Feature Redirection Map) + Analysis 7 (Feature Stability)
Days 6-7:   Analysis 5 (RepE / Refusal Direction Trajectories)
Days 8-9:   Analysis 4 (Attention Forensics) — only if earlier analyses show structure
Days 10-11: Analysis 6 (Activation Patching) — only if attention forensics succeeded
Days 12-13: Write up + figures
```

---

## Analysis 1: Layer Sweep for Probe Placement

**Priority**: FIRST. Foundational. Lowest effort.

**Inputs**: All 15 rounds of stealth_hard_s42 + control arm (once complete). Extract victim hidden states at layers 0, 4, 8, 12, 16, 20, 24, 28, 31. Use last token position before each victim response. Both wins and losses.

**Method**:
1. For each layer, train logistic regression probe (L2, balanced weights, 5-fold stratified CV)
2. Record AUC, precision, recall, F1 per layer
3. Plot AUC vs layer number

**Positive result**: Sharp peak at one or a few layers → safety checkpoint identified. If peak differs from probe training layer → stealth may be routing around wrong layer.

**Null result**: Uniform AUC > 0.90 everywhere → safety info is distributed, weakens "localized vulnerability" narrative.

**Time**: ~1 day.

---

## Analysis 2: Logit Lens Turn-by-Turn

**Priority**: SECOND. Highest novelty — this figure doesn't exist in the literature for multi-turn jailbreaks.

**Inputs**: 30-50 stealth wins, 30-50 control wins, 50 failures. Full conversation text for replay.

**Method**:
1. Load Llama-3.1-8B-Instruct into TransformerLens
2. For each conversation, forward pass at each turn (5 per conversation)
3. At each layer (every 4th), apply unembedding matrix. Compute softmax.
4. Define compliance tokens: {"Sure", "Here", "Step", "Okay", "Certainly"} and refusal tokens: {"Sorry", "cannot", "can't", "unable", "apologize"}
5. Compute P(compliance) - P(refusal) at each (layer, turn)
6. Average within condition. Produce 2D heatmap (turn × layer × compliance delta)

**Positive result**: Clear "flip point" at specific layer-turn. Stealth wins flip differently than control wins (different layer, or gradual vs abrupt).

**Null result**: Identical heatmaps for stealth and control → same mechanism, stealth just does it better.

**Failure modes**:
- Logit lens less informative at early layers with RMSNorm → use tuned lens as backup
- 5-turn conversations can be 2000+ tokens → budget 40GB+ VRAM
- "I" appears in both compliance and refusal sets → use full distribution, not just specific tokens

**Time**: 2-3 days.

---

## Analysis 3: Stealth Feature Redirection Map

**Priority**: THIRD. Most direct test of stealth mechanism.

**Inputs**: Successful jailbreaks from stealth arm (~279 wins) and control arm. Hidden states at peak layer from Analysis 1.

**Method**:
1. Extract SAE feature activations at breach turn for stealth wins and control wins
2. Compute mean activation vector per condition
3. Compute difference: delta = mean(stealth) - mean(control)
4. Identify top suppressed features (negative delta) and top amplified features (positive delta)
5. Cross-reference suppressed features with probe's top features → is stealth targeting the probe?
6. Cross-reference amplified features with SAE feature labels (Neuronpedia/SAELens)
7. Multiple comparison correction (Benjamini-Hochberg)

**Positive result**: Strong overlap between probe features and stealth-suppressed features AND interpretable amplified features (hypothetical framing, academic discussion, role-play).

**Null result**: Random overlap, small effect sizes after correction.

**Failure mode**: SAE dictionaries may not have pretrained SAEs for Llama-3.1-8B. Check SAELens availability before starting.

**Time**: 1-2 days.

---

## Analysis 5: RepE / Refusal Direction Trajectories

**Priority**: FOURTH (skip to this after Analysis 3, before Attention). Strong complement to feature redirection.

**Inputs**: 100+ refusal conversations, 100+ compliance conversations (for contrastive direction extraction). Then stealth wins, control wins, and failures.

**Method**:
1. Extract refusal direction via Arditi et al.: mean difference in residual stream between refusing and complying responses at peak safety layer
2. For each conversation, project victim hidden state at each turn onto refusal direction → scalar "refusal score" per turn
3. Plot trajectories: turn (1-5) vs refusal projection. Mean + CI per condition.
4. Compare trajectory shapes: gradual erosion (stealth) vs abrupt flip (control)?

**Positive result**: Stealth = smooth decrease (persuasion). Control = abrupt drop (overwhelming). Two mechanistically distinct jailbreak strategies.

**Null result**: Same trajectory shape for both conditions.

**Note**: Partially implemented in `emotional_ai/turnstile/directions.py`. Adapt existing code.

**Time**: 2-3 days.

---

## Analysis 4: Attention Forensics (STRETCH)

**Priority**: Fifth. Strongest weak-to-strong evidence but highest noise risk.

**Inputs**: 50 cleanest stealth wins (highest JBB confidence). 50 matched failures.

**Method**:
1. Forward pass through victim model at breach turn
2. Extract attention weights from all heads at first compliance token
3. Map attention back to all prior context tokens. Annotate by turn and speaker.
4. Compute: fraction of attention to adversary tokens vs victim tokens per head
5. Compare wins vs failures

**Positive result**: Adversary tokens from early turns get anomalously high attention at compliance moment → 3B learned to exploit 8B's attention structure through black-box interaction.

**Honest assessment**: Most likely to produce noise. Attention is noisy, positional confounds are strong. Treat as bonus.

**Time**: 2-3 days.

---

## Analysis 6: Activation Patching (STRETCH)

**Priority**: Sixth. Deepest finding but most compute-expensive.

**Inputs**: 20-30 successful jailbreaks + matched "clean" counterfactuals where victim refuses.

**Method**: Standard causal tracing. At each (layer, token) on breach turn, patch activation from clean run. Measure compliance probability change.

**Compute warning**: 32 layers × 500 tokens × 30 conversations = 480,000 forward passes. Even subsampled (every 4th layer, every 10th token), this is significant GPU time. Consider patching ONLY at adversary's key tokens (from Analysis 4) at peak safety layer.

**Time**: 3-5 days.

---

## Analysis 7: Cross-Round Feature Stability

**Priority**: Can run in parallel with Analysis 3.

**Inputs**: All 15 rounds of stealth_hard_s42.

**Method**:
1. Per round, extract SAE features at breach turn for all wins
2. Identify top-k (k=20, not 50 — small sample per round) most active features
3. Compute Jaccard similarity between feature sets across all round pairs
4. Plot heatmap: round × round

**Positive result**: Low turnover = architectural vulnerabilities (patchable). High turnover = cat-and-mouse (not patchable).

**Warning**: Only 8-30 wins per round. Use k=20 and consider aggregating adjacent rounds.

**Time**: 1-2 days.

---

## Paper Structure

**Title candidates** (avoid "Weak-to-Strong Jailbreaking" — Zhao et al. ICML 2025 already owns this term):
- "Safety Representations Predict but Do Not Prevent Jailbreaks: Mechanistic Evidence from Cross-Scale Adversarial Self-Play"
- "The Victim Knows: How Stealth Adversaries Exploit the Gap Between Safety Representations and Safety Behavior"
- "Cross-Scale Adversarial Steering: A 3B Model Learns to Route Around an 8B Model's Safety Representations"

**Sections**:
1. **Introduction**: Weak-to-strong failure is a concrete alignment concern. We demonstrate it empirically and explain it mechanistically.
2. **Method**: DPO self-play, safety probe, stealth objective, hardened victim.
3. **The victim knows** (probe results): AUC 0.96. Layer sweep. Logit lens heatmap.
4. **Stealth exploits the knowledge-behavior gap** (behavioral + mechanistic): Stealth ASR > control. Feature redirection map. RepE trajectories.
5. **The adversary learns the victim's architecture** (if Analysis 4/6 work): Attention forensics, activation patching.
6. **Safety vulnerabilities are localized** (if Analyses 1+6 work): Small subspace, patchable.
7. **Discussion**: Defense implications, limitations, future work.

**Minimum viable paper** (if results are mixed): Analyses 1 + 2 + behavioral finding. "Safety Representations Predict but Don't Prevent Jailbreaks." Workshop-level (NeurIPS SoLaR, SafeGenAI).

**Medium paper** (Analyses 1-3 + 5): Adds "where does safety live?" and "how does stealth route around it?" SaTML 2027.

**Full paper** (all 7): Competitive for ICLR 2027.

---

## Key Warnings

1. **Naming collision**: Zhao et al. (ICML 2025) already own "Weak-to-Strong Jailbreaking" as a paper title. Their method (logprob arithmetic, 99% ASR) is totally different. Use different terminology in the paper — "cross-scale adversarial steering" or "subversion." The repo name is fine but don't use it as the paper title.

2. **ASR comparison**: 15-25% ASR will be compared unfavorably to GOAT (97%), PAIR, Zhao et al. (99%). Must articulate why constrained setting (small learned adversary, no victim logits, DPO only) is the RIGHT setting for mechanistic insight.

3. **Judge reliability**: Documented history of judge false positives (Llama Guard 100% FP rate). Consider human-verifying a sample of the wins used for mechanistic analyses. If attention forensics runs on a false positive, it's analyzing noise.

4. **SAE availability**: Check whether SAELens has pretrained SAEs for Llama-3.1-8B before starting Analysis 3. If not, you'll need to train your own or use a different model.

---

## Key References

- Zhao et al. (ICML 2025) — "Weak-to-Strong Jailbreaking" (logprob method, different from ours)
- Arditi et al. (2024) — "Refusal in Language Models Is Mediated by a Single Direction"
- JailbreakLens (He et al. 2024) — representation + circuit analysis of jailbreaks
- Latent Sentinel (ICLR 2026) — layer-wise probes for jailbreak detection
- Self-RedTeam (Allen AI) — self-play red-teaming (no interp)
- "Knowing without Acting" (2026) — disentangled geometry of safety mechanisms
- ASGUARD (ICLR 2026) — safety enforced by specific attention heads
- Anthropic Attribution Graphs — multi-step safety feature construction
