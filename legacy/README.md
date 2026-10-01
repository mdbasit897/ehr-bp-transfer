# Legacy: preliminary analysis (superseded)

This folder contains the preliminary analysis reported in arXiv:2507.19530 (version 1) and
submitted to IEEE BIBM 2026: the original script `cross_institutional_bp_validation.py` and
its original outputs (`outputs/`).

**It is superseded by the analysis in `../revision/` and must not be used.** Its main
conclusion (negative external R² attributed to "fundamental cross-institutional barriers")
was caused by errors in the pipeline. Most importantly, the eICU blood-pressure target was
taken from arterial-line columns while the MIMIC-III target was mostly cuff pressure. All
errors are listed in `../docs/REVISION_CHANGES.md`, section A, and discussed in the paper.

It is kept only so that the preliminary results can be traced. The BigQuery project ID was
replaced by a placeholder (`your-gcp-project-id`).
