## cot-survival

Do activation probes still catch reward hacking after a model learns to hide it from a chain-of-thought monitor?

### basic question

If you put a chain-of-thought (CoT) monitor in the RL reward, models learn to keep reward hacking while removing the evidence from their reasoning text (Baker et al.). Activation probes are one proposed fix, since they read internal state rather than text. But we already know a model can learn to evade a probe when the probe itself is in the reward (RL-Obfuscation; Bailey et al., ICLR 2026).

The case nobody has measured is the indirect one. The model is optimized against the CoT monitor only, and the probe stays frozen and never enters the reward. Does hiding hacking from the text also hide it from the activations? RL-Obfuscation lists this indirect setting as future work.

### Setup
Policy: Qwen2.5-Coder-1.5B-Instruct, trained with GRPO and LoRA.
Task: coding problems where the reward checks only the visible tests, so special-casing them is a reachable hack. Hidden tests (EvalPlus MBPP+) give the true label, independent of any monitor.

### Training arms:
- Task reward only, so hacking emerges.
- Task reward minus a CoT monitor penalty, which puts pressure on the text.
- Task reward minus a probe penalty, a positive control showing the probe can be evaded when targeted directly.
- Monitors, scored on every checkpoint:
  - an LLM judge that reads the CoT;
  - linear probes frozen before training starts;
  - probes refit at each checkpoint, which separates drift (recalibrate and it works again) from the information actually being removed.

### What we measure
True hacking rate (from hidden tests) against training steps, in each arm.
Recall of each monitor at 1% and 5% false positive rate, not just AUROC.
Whether the frozen probe holds up in arm 2 while the CoT monitor fails.

### What counts as a result
Frozen probe survives CoT pressure: latent monitors are a real backstop when CoT monitoring breaks.
Frozen probe fails, refit probe works: the representation drifted but the signal is still there, so probes need recalibration during training.
Both fail: text pressure removes the signal from the activations too, which undercuts using probes as a fallback.

Every one of these outcomes is worth reporting.

First gate: hacking has to emerge in arm 1 at 1.5B parameters. If it does not, scale up to 3B or use resample-until-hack (Li et al.) before running anything else.
