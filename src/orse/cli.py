import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from .common import load, save, table, plot, classification, auc, roc
from .core import adapter, probe, fuse


class Source(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channels, 32, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.model(x)


def prepare(a, c):
    manifest = load(a.input)
    groups = {}
    records = []
    for record in manifest:
        if record["group"] in groups and groups[record["group"]] != record["split"]:
            raise ValueError("Physical bearing/run group crosses split")
        groups[record["group"]] = record["split"]
        path = Path(record["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        records.append(dict(record, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", records)


def train(a, c):
    torch.manual_seed(c["seed"])
    records = [
        r
        for r in load(Path(a.data) / "manifest.json")
        if r["split"] == "train" and r["source"] == a.source
    ]
    if not records:
        raise ValueError("No matching source training records")
    classes = max(r["label"] for r in records) + 1
    model = Source(2 if a.adapter == "waveform_speed" else 1, classes).to(c["device"])
    opt = torch.optim.Adam(model.parameters(), lr=c["source_lr"])
    rows = []
    for epoch in range(c["source_epochs"]):
        total = 0.0
        for record in records:
            data = np.load(record["path"])
            value = adapter(
                data["x"],
                data["speed"],
                float(data["sample_rate"]),
                a.adapter,
                a.length,
            )
            loss = nn.functional.cross_entropy(
                model(torch.from_numpy(value)[None].to(c["device"])),
                torch.tensor([record["label"]], device=c["device"]),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        rows.append(dict(epoch=epoch + 1, loss=total / len(records)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(
        model.cpu().eval(),
        torch.zeros(1, 2 if a.adapter == "waveform_speed" else 1, a.length),
    )
    scripted.save(str(output / "source.ts"))
    save(
        output / "source.json",
        dict(
            path=str((output / "source.ts").resolve()),
            adapter=a.adapter,
            length=a.length,
            classes=classes,
        ),
    )
    table(output / "history.csv", rows)


def load_models(a, c):
    settings = load(a.models)
    models = []
    for s in settings:
        path = Path(s["path"])
        path = path if path.is_absolute() else Path(a.models).parent / path
        models.append((torch.jit.load(str(path), map_location=c["device"]).eval(), s))
    return models


def calibrate(a, c):
    models = load_models(a, c)
    records = [
        r for r in load(Path(a.data) / "manifest.json") if r["split"] == "calibration"
    ]
    raw = [probe(r, models, c) for r in records]
    if not raw:
        raise ValueError("No unlabeled calibration records")
    margins = np.stack([1 - r[1] for r in raw])
    disagreements = np.stack([r[2] for r in raw])
    calibration = {"energy_center": 0.0}
    for key, values in [("margin", margins), ("disagreement", disagreements)]:
        med = np.median(values, axis=0)
        mad = np.median(np.abs(values - med), axis=0)
        calibration[key + "_median"] = med.tolist()
        calibration[key + "_mad"] = np.maximum(mad, 1e-3).tolist()
    energy = [fuse(z, m, d, c, calibration, a.variant)[3] for z, m, d, n in raw]
    calibration["energy_center"] = float(np.median(energy))
    risks = [fuse(z, m, d, c, calibration, a.variant)[1] for z, m, d, n in raw]
    calibration["threshold"] = float(np.quantile(risks, c["coverage"]))
    calibration["variant"] = a.variant
    save(Path(a.output) / "calibration.json", calibration)


def evaluate(a, c, variant):
    models = load_models(a, c)
    calibration = load(a.calibration)
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "test"]
    rows = []
    probabilities = []
    risks = []
    source_risks = []
    errors = []
    for record in records:
        z, m, d, count = probe(record, models, c)
        posterior, rejection, q, energy = fuse(z, m, d, c, calibration, variant)
        k = len(posterior)
        label = record["label"]
        prediction = int(posterior.argmax())
        rejected = variant != "no_rejection" and rejection > calibration["threshold"]
        rows.append(
            dict(
                id=record["id"],
                label=label,
                prediction=k if rejected else prediction,
                known_prediction=prediction,
                risk=rejection,
                accepted=not rejected,
                orbit_size=count,
            )
        )
        probabilities.append(posterior)
        risks.append(rejection)
        source_risks.extend(q.tolist())
        errors.extend((z.argmax(1) != label).tolist())
    y = np.array([r["label"] for r in records])
    pp = np.stack(probabilities)
    k = pp.shape[1]
    unknown = y < 0
    y_open = np.where(unknown, k, y)
    pred = np.array([r["prediction"] for r in rows])
    one_hot = np.eye(k + 1)[pred]
    fpr, tpr = roc(unknown, risks)
    summary = dict(
        method=variant,
        **classification(y_open, one_hot, np.array(risks)),
        unknown_auroc=auc(unknown, risks),
        fpr95=float(fpr[np.flatnonzero(tpr >= 0.95)[0]]) if fpr is not None else None,
        source_error_correlation=float(spearmanr(source_risks, errors).statistic),
        coverage=np.mean([r["accepted"] for r in rows]),
    )
    correct = (pp.argmax(1) == y) & ~unknown
    ordered = np.argsort(risks, kind="stable")
    summary["aurc"] = float(
        np.mean(np.cumsum(~correct[ordered]) / np.arange(1, len(y) + 1))
    )
    output = Path(a.output) / variant
    table(output / "predictions.csv", rows)
    save(output / "metrics.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["prepare", "train-source", "calibrate", "run", "compare", "plot"],
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/vibration")
    p.add_argument("--output", default="runs")
    p.add_argument("--models", default="models.json")
    p.add_argument("--calibration", default="checkpoints/calibration.json")
    p.add_argument("--source", default="source0")
    p.add_argument(
        "--adapter",
        choices=["waveform", "spectrum", "order", "waveform_speed"],
        default="waveform",
    )
    p.add_argument("--length", type=int, default=1024)
    p.add_argument(
        "--variant",
        choices=["orse", "uniform", "entropy", "margin", "no_rejection"],
        default="orse",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train-source":
        train(a, c)
    elif a.command == "calibrate":
        calibrate(a, c)
    elif a.command == "run":
        evaluate(a, c, a.variant)
    elif a.command == "compare":
        rows = [
            evaluate(a, c, m)
            for m in ["uniform", "entropy", "margin", "no_rejection", "orse"]
        ]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
