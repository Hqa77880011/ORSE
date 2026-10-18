"""Bearing vibration from angular impact trains and source-specific resonances."""

import argparse
import json
from pathlib import Path

import numpy as np


def vibration(rng, fault, source, samples=2048, sample_rate=2000):
    time = np.arange(samples) / sample_rate
    base = [22, 31, 27][source]
    speed = base * (1 + rng.uniform(0.03, 0.12) * np.sin(2 * np.pi * 0.7 * time + rng.uniform(0, 6.28)))
    angle = 2 * np.pi * np.cumsum(speed) / sample_rate
    order = [0, 3.6, 5.1, 7.7][fault]
    impacts = np.zeros(samples)
    if order:
        phase = angle * order / (2 * np.pi) + rng.random()
        events = np.flatnonzero(np.diff(np.floor(phase), prepend=np.floor(phase[0])) > 0)
        impacts[events] = rng.uniform(0.5, 1.2, len(events))
    resonance = [260, 390, 325][source] + rng.uniform(-20, 20)
    kernel_time = np.arange(160) / sample_rate
    kernel = np.exp(-rng.uniform(65, 110) * kernel_time) * np.sin(2 * np.pi * resonance * kernel_time)
    signal = np.convolve(impacts, kernel, mode="full")[:samples]
    signal += rng.uniform(0.1, 0.25) * np.sin(angle) + 0.05 * np.sin(2 * angle + rng.random())
    signal += rng.normal(0, rng.uniform(0.06, 0.15), samples)
    return dict(x=signal.astype(np.float32), speed=speed.astype(np.float32),
                sample_rate=np.float32(sample_rate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 10:
        parser.error("Use at least ten independent bearing/run groups")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        split = "train" if group < 0.6 * args.groups else "calibration" if group < 0.8 * args.groups else "test"
        source = group % 2 if split == "train" else 2
        for fault in range(3 if split == "train" else 4):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, group, fault]))
            name = f"bearing_{group:03d}_{fault}.npz"
            np.savez_compressed(output / name, **vibration(rng, fault, source))
            records.append(dict(id=f"{group}:{fault}", path=name, group=f"bearing-run-{group}",
                                split=split, source=f"source{source}" if source < 2 else "target",
                                label=fault if fault < 3 else -1))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="angular impact process convolved with damped resonant response",
        truth="normal/known fault-order families; order 7.7 absent from source training",
        seed=args.seed, groups=args.groups, speed_units="revolutions/second",
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
