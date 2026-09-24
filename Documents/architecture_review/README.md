# HATI architecture and implementation review

Prepared on 24 September 2026 for scientific and engineering feedback.

- **Share the PDF:** [31-page technical dossier](HATI_architecture_and_implementation_review.pdf), with the HATI logo, equations, architecture diagrams, three recorded hazard maps, the observability map, control results and references.
- **Share editable text:** [Expanded Markdown](HATI_review_for_feedback.md), with immutable code links and relative figure references. Include the `figures` directory when sharing this format outside the repository.
- **Edit the source:** [Source Markdown](HATI_architecture_and_implementation_review.md). The builder expands its code-reference marker into the review copy.
- **Inspect provenance:** [Frozen evidence snapshot](evidence_snapshot.json). This contains recorded metrics, configurations, archive identity, source hashes and software-check summaries.

The implementation snapshot is commit `40cf64075c331e19d7fb57c4b2b4b3b4de17be66` (core package version 2.5.5 with the experimental adaptive extension). Full workstation results belong to the earlier commit `8347e6dbfc7532bb59c3311ff888ec3f7f9efd9a`. The document distinguishes these results from the local adaptive checks. A completed full adaptive workstation campaign was not available when this dossier was prepared.

Suggested improvements and reviewer questions are proposals. The document makes no claim of validated polar detection performance, calibrated hazard probabilities, 10 cm measurement accuracy, landing clearance or demonstrated reduction in mission loss.

## Rebuild

Install the document dependencies in your chosen Python environment:

```sh
python -m pip install reportlab matplotlib numpy Pillow
python Documents/architecture_review/build_dossier.py --reuse-figures
```

The command runs from the repository root and uses the committed figures and evidence snapshot. It rebuilds the PDF, equation images and expanded Markdown. It uses Calibri on Windows when available and DejaVu fonts otherwise; another font may alter page layout, so inspect the rendered PDF after rebuilding.

To replot the archived raster figure, also install `rasterio` and supply a directory containing `terrain_hazard.tif`, `shadow_hazard.tif`, `fused_hazard.tif` and `fusion_status.tif` from the map stage of `athena-watch-01_results.zip`:

```sh
python Documents/architecture_review/build_dossier.py --raster-dir /path/to/saved/maps
```

The builder uses saved measurements. It never invokes HATI inference or ISIS ingestion. Layout inspection metadata is written under `tmp/architecture_review/`.

Section 26 supplies twelve specific review questions and a suggested feedback format. Reviewers should identify the affected section or code reference, the concern and consequence, supporting evidence, a proposed test and its priority.
