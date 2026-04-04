# Weak-to-Strong Jailbreaking

Mechanistic interpretability of cross-scale adversarial steering: how a 3B-parameter model learns to jailbreak an 8B-parameter model, and what happens inside the victim.

## Research Question

Can a less-capable model reliably steer a more-capable model into unsafe behavior? And if so, *what mechanistic pathway does it exploit?*

We study this using a multi-turn adversarial self-play setup ([Turnstile](https://github.com/kilojoules/turnstile)) where a DPO-trained 3B adversary jailbreaks a frozen or hardened 8B victim over 5-turn conversations. A "stealth" variant trains the adversary to simultaneously evade a linear probe on the victim's hidden states.

## Key Finding (from Turnstile)

Against a **frozen** (static) victim, stealth doesn't help — both adversaries achieve ~26% ASR. Against a **hardened** (adaptive) victim, stealth-trained adversaries maintain 21-25% ASR while controls collapse to 11-15%. The victim learns to refuse detectable attacks but can't learn to refuse stealth attacks because they don't trigger the safety representations the victim trains on.

This project goes deeper: *what is the stealth adversary actually doing inside the victim's representations?*

## Analyses

Using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [SAELens](https://github.com/jbloomAus/SAELens) on conversation data from the Turnstile stealth experiment.

### Tier 1 (core)

| # | Analysis | Question | Status |
|---|----------|----------|--------|
| 1 | Stealth feature redirection map | Which SAE features does stealth suppress vs amplify? Has it found alternative compliance routes? | Planned |
| 2 | Layer sweep | Is the probe watching the right layer? Where does jailbreak detection AUC peak? | Planned |
| 3 | Logit lens turn-by-turn | At which layer and turn does the safety decision flip? Gradual corruption or checkpoint flip? | Planned |
| 4 | Attention forensics at breach turn | Has the 3B adversary learned to exploit the 8B's attention architecture? | Planned |
| 5 | Activation patching at breach turn | Which (layer, token) positions are causally responsible for compliance? | Planned |
| 6 | RepE / refusal direction trajectories | Does jailbreak work by destroying the refusal signal or activating a competing drive? | Planned |
| 7 | Cross-round feature stability | Are vulnerabilities architectural (fixed) or strategic (shifting)? | Planned |

### Tier 2 (ambitious)

| # | Analysis | Question | Status |
|---|----------|----------|--------|
| 8 | Mutual information: adversary-victim hidden states | Does stealth training improve the adversary's implicit model of the victim? | Planned |
| 9 | Differential propagation (cryptanalysis-inspired) | Does stealth route the causal signal through unmonitored layers? | Planned |
| 10 | Phase transition detection (Fisher information) | Is there a tipping point where compliance becomes self-sustaining? | Planned |

## Setup

- **Adversary**: Llama-3.2-3B-Instruct + LoRA, trained with DPO on win/loss pairs
- **Victim**: Llama-3.1-8B-Instruct (frozen or hardened via safety fine-tuning on refusal data)
- **Goals**: JailbreakBench 100 standardized harmful behaviors
- **Judge**: Llama-3.1-70B-Instruct (dual-judge with Llama Guard)
- **Conversations**: 5-turn, 100 per round, 13+ rounds per seed

## Data

`data/stealth_hard_s42/` — complete stealth experiment (seed 42, hardened victim, 13 rounds, 100 conversations/round).

## Dependencies

```
transformerlens
sae-lens
torch
einops
```

## Related Repos

- [REDKWEEN](https://github.com/kilojoules/REDKWEEN) — automated red-teaming pipeline with SAE probes
- [Turnstile](https://github.com/kilojoules/turnstile) — multi-turn adversarial self-play (stealth A/B experiment)
- [Silent Killers](https://github.com/kilojoules/silent-killers) — LLM exception-handling audit (under review at COLM 2026)
