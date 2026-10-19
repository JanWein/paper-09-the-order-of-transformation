# Reproducibility

Paper 09: The Order of Transformation:: Learning, Remediation, and the Option to Stop

Run `python3 reproduce.py` from the repository root after installing `requirements.txt`.
It invokes these original scientific entry points in order:

```sh
python3 code/analysis.py
python3 code/rational_learning.py
python3 code/cost_regions.py
```

The wrapper stops on the first failed script. Scientific scripts are copied byte-for-byte from the complete research package; only the repository wrappers and documentation are new. Regeneration overwrites generated result tables and figures in the working tree. Retain a clean checkout or use a separate clone when comparing with the supplied outputs.

The current paper's LaTeX and supplied PDF are byte-identical to the checked F1 edition. Equations, proofs and scientific results were not rewritten for GitHub. Earlier internal review correspondence, chat references, editorial QA screenshots, build logs and full nested working archives are excluded. Supplementary scientific material required by the current article or reproduction is retained.

Some historical result metadata describe earlier scientific revisions. Those files are retained as recorded scientific outputs; editorial F1 changed the abstract and presentation. Runtime timestamps, environment metadata and floating-point display details can vary. See exact arithmetic assertions and stated numerical tolerances in each script.

`output/rational_policy_grid.csv` is a supplementary numerical grid; the exact cost-region script and analytical proofs establish the stated boundary results. `code/analysis.py` is a nonstructural synthetic estimator stress test.

Default Python entry points are the executed reference. Optional R files and unrun real-data estimator interfaces are documented separately and are not presented as executed empirical analyses.
