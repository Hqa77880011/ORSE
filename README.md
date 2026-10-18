# ORSE

Implementation of order-preserving counterfactual risk for selective multi-source-free fault diagnosis.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## Data and source models

Download vibration records from [CWRU](https://engineering.case.edu/bearingdatacenter), [Paderborn](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter) or JNU through their dataset providers. Segment only after grouping by physical bearing, acquisition run, machine and condition. Each NPZ contains one-dimensional `x`, scalar or sample-wise `speed` in revolutions/second, and scalar `sample_rate` in Hz. A JSON manifest lists `id`, `path`, `group`, `split` (train/calibration/test), `source` and `label`. Unknown test faults use label -1; all known-source models share class order. Calibration labels are never read by the calibrator.

```bash
orse prepare --input data/manifest.json --data data/vibration
orse train-source --data data/vibration --source source0 --adapter waveform --output checkpoints/source0
```

Train each source separately or provide frozen TorchScript models with native adapters. `models.json` is a list of `{path, adapter, length, classes}` using the generated `source.json` contents. Relative model paths resolve against this file. Each model returns logits [batch,known-class]. Supported native paths are waveform, spectrum, order and waveform plus speed. Waveform adapters preserve the native sampling grid, truncating or zero-padding to the trained window length; they do not undo the physical time scaling by resampling it away. The bundled source CNN is a direct trainable baseline; externally trained source models retain their native input preprocessing by choosing the corresponding adapter.

```bash
orse calibrate --data data/vibration --models models.json --output checkpoints
orse run --data data/vibration --models models.json --calibration checkpoints/calibration.json --output runs
orse compare --data data/vibration --models models.json --calibration checkpoints/calibration.json --output runs
orse plot --output runs
```

## Physical orbit and fusion

Time scaling resamples the waveform at `a*t` and speed at `a*omega(a*t)`, retaining the common angular horizon. A normalized angular-resampled spectrum screens candidates before native model evaluation. The gate uses robustly normalized `1 - mean margin` plus mean Jensen–Shannon disagreement. Per-source median/MAD and the fused energy center are frozen on unlabeled calibration data. The post-fusion entropy/positive energy score sets an empirical coverage quantile. Unknown labels affect only metrics.

Controls include uniform fusion, entropy weighting, margin-only weighting and removal of rejection. For a speed-permutation control, prepare a separate manifest/NPZ set with shuffled speed records within the locked split while preserving waveform IDs, and rerun calibration and evaluation. To study unreliable-source prevalence, modify the frozen source pool using independently trained noisy-label or withheld-speed sources. External source-free adaptation algorithms are separate source-pool alternatives, rather than aliases of these gates.

## Parameters and metrics

`configs/default.json` specifies scale grid, order bins/tolerance, disagreement coefficient, softmax temperature, entropy weight, calibration coverage, source training and device. Source training writes final `source.ts`, adapter metadata and loss history. Evaluation writes original/report predictions, acceptance, rejection score and accepted orbit size. `metrics.json` includes open-set macro-F1, unknown AUROC/FPR95, known-decision AURC (unknowns count as errors), source-risk/error Spearman correlation and report coverage. `predictions.csv` permits threshold and source-pool analysis. Rates use [0,1]; comparison outputs and PNG/SVG plots are computed from run files. For controlled comparisons use one prespecified threshold, or calibrate each variant separately before comparing its operating coverage; `compare` uses the supplied shared calibration.

## Registered compact reproduction setting

Locally trained source CNNs with waveform adapters, fixed 1,024-sample windows, 30 source epochs and the registered five-scale physical orbit. The source trainer supplies weights; no externally pretrained model is necessary.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
orse prepare --config configs/reproduce.json --input data/manifest.json --data data/reproduce
orse train-source --config configs/reproduce.json --data data/reproduce --source source0 --adapter waveform --length 1024 --output checkpoints/source0
orse train-source --config configs/reproduce.json --data data/reproduce --source source1 --adapter waveform --length 1024 --output checkpoints/source1
orse pool --config configs/reproduce.json --source-metadata checkpoints/source0/source.json checkpoints/source1/source.json --models models.json
orse calibrate --config configs/reproduce.json --data data/reproduce --models models.json --output checkpoints/reproduce
orse compare --config configs/reproduce.json --data data/reproduce --models models.json --calibration checkpoints/reproduce/calibration.json --output runs/reproduce
```

`pool` assembles the two actually trained source metadata files and checks equal class counts; source0/source1 training records must use the same registered class order.

Required external inputs: bearing/run-disjoint vibration with actual speed and sample rate, source/fault partitions. The formal input fields are specified above; preparation/training interfaces consume those fields directly. No particular paper dataset or pretrained task model is assumed beyond these declared inputs.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
orse-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
