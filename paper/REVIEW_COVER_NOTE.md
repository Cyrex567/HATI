# HATI: review request (cover note)

Hi. I'm Gergő Csaba Morvai, an engineering undergraduate on the Máni lunar-mission science team. HATI is an independent, interpretable terrain-hazard pipeline I built to test whether pre-landing orbital data could have flagged the sub-resolution hazard behind the 2025 IM-2 "Athena" tip-over. The attached report is an honest status and roadmap: what is proven on real data, what is still only synthetic, and what I plan to do next. I'm sending it before committing serious compute to the real-data run, because a hard critique now is worth more to me than a polished result later.

**What I'd value most from the review:**
- Whether the shadow-kinematics estimator is biased in a way my synthetic benchmark hides, especially shadows that fall across real slopes rather than flat ground.
- Whether noisy-OR is the right way to fuse a deterministic gate with a probabilistic obstacle layer without double-counting correlated evidence.
- Whether a resolution ladder on degraded orbital data is a fair proxy for real detection performance, or whether anti-aliased degradation flatters the method.
- Whether cross-site transfer at AUC ≈ 0.8 is strong enough to anchor a safety-critical gate, or simply is not good enough.
- What ground truth would make a *calibrated* strike probability defensible, given how few instrumented lunar landing outcomes exist.

The fastest way in is the evidence ledger (Table 1) and the closing "Questions we would most like pressed" section. I have tried to make it impossible to mistake what is established for what is planned.

**Attached for depth (optional):** the v2.0 architecture paper and the shadow-kinematics method note.

Thanks for taking the time. Please tear into it.

Best,
Gergő Csaba Morvai
