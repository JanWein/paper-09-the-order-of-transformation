# The Order of Transformation:: Learning, Remediation, and the Option to Stop

**Dr. Jan-Hendrik Weinert** · Working paper · Editorial edition F1, 19 October 2026

[Read the paper](paper.pdf) · [LaTeX source](manuscript.tex) · [Editable Markdown](manuscript.md) · [Reproduction instructions](REPRODUCIBILITY.md)

## Abstract

Implementation order trades off better diagnosis against the option to avoid sunk remediation costs. We solve a binary learning and entry-certificate model with endogenous production experimentation, direct remediation and no remediation. Under the stated information ordering and equal assigned pilot costs, tools-first dominates all alternatives exactly on an open physical-cost interval. Closed-form cutoffs identify its competitors and the pilot-cost ceiling for nonemptiness. Synthetic primitives yield the full certificate-fee partition, all ties and an explicit reset threshold at which the interval disappears. Known repair, an independent reset and revealing public validation retain their Bayesian nulls. The contribution is a model-specific cost and fee characterization, supported by complete proofs and executable checks; no firm data are analyzed.

## Contents

- `paper.pdf`: complete paper, including proofs, bibliography and appendices.
- `manuscript.tex`: authoritative mathematical typesetting source; `manuscript.md`: editable alternative.
- Calculation scripts, recorded synthetic outputs, and the paper-specific source ledgers and prospective research protocols.
- `reproduce.py`, `requirements.txt`: one entry point and dependencies for the supplied analysis.
- `build_paper.sh`: rebuilds the current PDF from the LaTeX source without replacing the supplied PDF.
- `CITATION.cff`: citation metadata; `SHA256SUMS.txt`: published payload checksums.

`output/rational_policy_grid.csv` is a supplementary numerical grid; the exact cost-region script and analytical proofs establish the stated boundary results. `code/analysis.py` is a nonstructural synthetic estimator stress test.

## Reproduce the calculations

Use Python 3.12. From the repository root, create a virtual environment and run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 reproduce.py
```

The analysis uses synthetic inputs or exact model primitives supplied with the code. It requires no private organizational data, API credentials or data download. Installing dependencies may require access to a package index. Full analytical proofs remain in the paper; finite computations supplement those proofs.

## Rebuild the paper

With TeX Live and the packages declared in `manuscript.tex` installed:

```sh
bash build_paper.sh
```

The result is `build/manuscript.pdf`. Compile from the repository root so figure paths resolve. The supplied `paper.pdf` is the visually checked publication edition. PDF byte hashes can vary between rebuilds because of timestamps and toolchain metadata. Do not regenerate this edition's LaTeX through Pandoc when a matching layout is required.

## Research status

This is a theoretical working paper with synthetic computational illustrations. Any empirical or field protocol included here is prospective, and no reported synthetic output is an observed firm result. Publication of this repository does not imply journal acceptance or empirical validation. The paper states its model-specific assumptions and limitations.

## Citation

Weinert, Jan-Hendrik (2026). *The Order of Transformation:: Learning, Remediation, and the Option to Stop*. Working paper, editorial edition F1.
Use the repository's citation metadata when citing or reusing the work.
