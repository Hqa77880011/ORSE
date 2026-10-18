# ORSE

Implementation of order-preserving counterfactual risk for selective multi-source-free fault diagnosis.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator creates angular-domain vibration workloads with independent acquisition groups, source identities and speed in revolutions/second. These are generated signals. Both frozen sources below are trained from their own declared source partitions.

```bash
python -m orse.synthetic --output data/generated --groups 30 --seed 17
orse prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Download vibration records from [CWRU](https://engineering.case.edu/bearingdatacenter), [Paderborn](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter) or JNU through their dataset providers. Segment only after grouping by physical bearing, acquisition run, machine and condition. Each NPZ contains one-dimensional `x`, scalar or sample-wise `speed` in revolutions/second, and scalar `sample_rate` in Hz. A JSON manifest lists `id`, `path`, `group`, `split` (train/calibration/test), `source` and `label`. Unknown test faults use label -1; all known-source models share class order. Calibration labels are never read by the calibrator.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
orse train-source --config configs/reproduce.json --data data/reproduce --source source0 --adapter waveform --length 1024 --output checkpoints/source0
orse train-source --config configs/reproduce.json --data data/reproduce --source source1 --adapter waveform --length 1024 --output checkpoints/source1
orse pool --config configs/reproduce.json --source-metadata checkpoints/source0/source.json checkpoints/source1/source.json --models models.json
orse calibrate --config configs/reproduce.json --data data/reproduce --models models.json --output checkpoints/reproduce
```

## Evaluation

```bash
orse compare --config configs/reproduce.json --data data/reproduce --models models.json --calibration checkpoints/reproduce/calibration.json --output runs/reproduce
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `source.json`, `history.csv`, `calibration.json`, `predictions.csv`, `metrics.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
