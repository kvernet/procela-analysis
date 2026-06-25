# Procela Analysis

<div align="center">

**Analysis and audit tools for Procela simulations**

[![PyPI version](https://img.shields.io/pypi/v/procela-analysis.svg)](https://pypi.org/project/procela-analysis/)
[![PyPI version](https://img.shields.io/pypi/v/procela.svg)](https://pypi.org/project/procela/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-procela.org-green.svg)](https://docs.procela.org)
[![CI](https://github.com/kvernet/procela-analysis/actions/workflows/ci.yaml/badge.svg)](https://github.com/kvernet/procela-analysis/actions/workflows/ci.yaml)

</div>

## Overview

Procela Analysis converts [Procela](https://github.com/kvernet/procela) simulation memory into structured
DataFrames and provides a suite of analytical tools for understanding
how competing mechanisms behave, when regimes shift, and whether
governance interventions succeed.

- **MemoryReader**: extract hypotheses, resolutions, and errors from
  Procela variables as pandas DataFrames
- **MechanismProfiler**: per-mechanism accuracy curves, falsifiability
  scores, and redundancy detection
- **MechanismEcology**: population-level analysis — dominance curves,
  niche overlap, extinction events, diversity indices, rank turnover
- **RegimeDetector**: unsupervised detection of structural breaks in
  mechanism error patterns
- **TransitionAnalyzer**: characterize what changes at regime boundaries
- **PolicyStability**: compare alternative resolution policies without
  simulating counterfactual trajectories
- **Epistemic signals**: rolling error, coverage, fragility,
  disagreement index, confidence spread
- **AuditReport**: self-contained interactive HTML report for
  publication supplementary materials
- **Visualization**: publication-ready matplotlib plots for all metrics

## Installation

```bash
pip install procela-analysis
```

## Quick Start

```python
from procela import Executive, Variable, RangeDomain, WeightedVotingPolicy
from procela_analysis import MemoryReader, MechanismEcology, AuditReport

# Run a Procela simulation
X = Variable("X", RangeDomain(0, 100), policy=WeightedVotingPolicy())
# ... add mechanisms, run simulation ...

# Read memory
reader = MemoryReader(X)
hypotheses = reader.hypotheses()
resolutions = reader.resolutions()
errors = reader.errors()

# Analyze mechanism competition
ecology = MechanismEcology(hypotheses, errors)
dominance = ecology.dominance_curve("X")
extinctions = ecology.extinction_events("X", threshold=0.05)

# Generate an audit report
report = AuditReport(hypotheses, resolutions, errors)
report.generate("audit.html")
```

## Development

```bash
git clone https://github.com/kvernet/procela-analysis.git
cd procela-analysis
python -m venv .venv
source .venv/bin/activate
make dev-install
make pre-commit
```

# Citation

If you use Procela Analysis in your research, please cite:

```bibtex
@software{procela_2026,
    title={Procela: Epistemic Governance in Mechanistic Simulations Under Structural Uncertainty},
    author={Kinson Vernet},
    year={2026},
    eprint={2604.00675},
    archivePrefix={arXiv},
    primaryClass={physics.comp-ph},
    url={https://arxiv.org/abs/2604.00675},
}
```

## License

Apache 2.0 — see [LICENSE](https://github.com/kvernet/procela-analysis).
