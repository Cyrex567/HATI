# HATI v2.5 scientific release assessment

10 September 2026. Core version 2.5.2. Branch `feat/v2.0-heatmap`.

HATI now includes an experimental image-domain shadow-root detector alongside the audited legacy pipeline. The scientific target is subpixel relief sensing poleward of 70 degrees. This release establishes software and synthetic evidence; it does not establish achieved lunar sensitivity, crater/boulder classification, a hazard probability or landing clearance. No ISIS application was executed on the Windows audit host.

## Implemented detector

`src/hati_core/shadow_likelihood.py` compares a static per-pixel image plus a brightness plane in every frame with the same nuisance model plus a moving shadow. On common observed pixels, an exact weighted projection removes static albedo and per-frame planes. Every illumination participates in the constrained shared-contrast fit. A contradictory frame reduces the joint improvement; it cannot silently abstain.

Directional darkening in the nuisance residual proposes roots. No connected-component minimum area, elongation test or width ceiling defines the final evidence. Circular support of radius six pixels limits fitting to the shadow origin, so later mergers outside this support cannot dominate the local fit. The render patch has radius twelve pixels and includes extra blur padding. Proposal competition and a finite candidate budget can still lose objects; `search_truncated` records the cap.

The template bank varies height and width independently, integrates fractional-pixel coverage, averages five approximate solar-disc strips and applies Gaussian optical/registration blur. Receiving-plane gradients modify the shadow intersection length. Subpixel root offsets sample minus one half, zero and plus one half pixel. These are forward-model parameters, not proof of a resolved object or a height confidence interval. Endpoints outside fitting support are flagged as censored.

The score is the square root of a constrained Gaussian likelihood improvement after nuisance removal, with a bounded positive shared amplitude. Results record per-frame fit changes, template identifiability, observed support, candidate cap, configuration hash and assumptions. Scores are rankings, not standard deviations measured in lunar data.

Optional Monte Carlo checks rerun the entire proposal/template search on independent Gaussian stacks with the same mask and supplied noise. The plus-one rank of the maximum includes location and template selection under that fixed complete-null model. It is not field calibration: unknown and spatially correlated noise, median normalization, registration and physical model mismatch remain outside the model.

Map bearings use the projection and affine Jacobian, tested for both poles and a rotated raster. `scripts/shadow_roots_real.py` reads audited Athena products without invoking ISIS; it is not a general multi-site ingestion system. One site geometry and receiving plane approximate the pilot.

## Additional corrections

The baseline audit details acquisition-cutoff enforcement, complete-download checks, processing contracts, NAC echo correction, retained ISIS labels, campt fail-closed behavior, registration sign/bound fixes, closure gates, corrected arc RMS, reference-grid output transforms, supersampled injection and configured-width recovery.

This release also makes diagonal-component connectivity consistent, assigns tied AUC scores half credit, and offers `Config(tri_baseline_m=...)` for a descriptor with a fixed physical sampling radius. TRI is opt-in to preserve historical scores; switching requires a new reference normalization. Historical AUC measures site separation and does not validate the changed descriptor or safety. `requirements-science.txt` installs the deterministic path without trained-model dependencies.

## Evidence and remaining limits

Evidence is in `tests/test_shadow_likelihood.py`, the earlier five offline suites and `Documents/v25_synthetic_benchmark.json`. The benchmark uses an independent tapered-caster renderer with 12-fold supersampling and an anisotropic PSF, not the detector's rectangular renderer. It covers static albedo, a 0.3 m high by 0.6 m wide caster, a metre-high caster, registration jitter and higher noise. Twelve seeds per scenario characterize a favorable seven-direction, 310-degree sweep on a 56-pixel synthetic scene. They do not establish sensitivity at Athena's available span or all terrain above 70 degrees.

Exactly aligned stationary albedo with brightness planes has zero fitted moving-shadow evidence in the deterministic regression. Noise produces nonzero selected maxima. Adding 0.25-pixel frame jitter creates stronger false candidates from fixed stains; broadening a template does not model those albedo-registration residuals. The benchmark reports this failure mode instead of assigning a field false-positive rate.

Necessary next validation steps remain: independent local registration checkpoints; shape-model verification and terrain ray tracing; native-sampling PSF/MTF characterization; physical albedo/occlusion nuisance models; correlated residual controls and held-out sites; injection before registration; independently annotated objects; size-dependent completeness and density intervals. Legacy crater/ridge injections remain simplified controls, not validated classification. Neither detector clears terrain when it observes no candidates.

## GPU host execution

Follow [WSL_AUDIT_RUN.md](WSL_AUDIT_RUN.md). The shell runner tests, queries, ingests and runs three assumed registration-blur scenarios. Actual ISIS compatibility, archive availability, gate yield and candidate stability require Linux execution. The search is CPU code; no CUDA performance claim is made.

## Scientific result worth pursuing

The first defensible result is a held-out object-level detection and false-discovery study stratified by size, effective native resolution, incidence, emission, sweep diversity and terrain. Use independent annotations with adjudication, separate calibration/test sites and spatial-block uncertainty intervals. Report misses and unknown regions. Size-frequency inference requires completeness and association uncertainty. Clearance also requires footprint-scale vehicle limits, navigation uncertainty and explicit non-observability.

The [baseline audit](AUDIT_2026-09-09.md) includes ranked findings, derivations, primary ISIS references and external validation resources. The branded report summarizes the audit and implementation. The physical route is promising and concrete software barriers are removed; a breakthrough in measured lunar performance remains a hypothesis to test.
