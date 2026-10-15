import numpy as np
from scipy.special import softmax, logsumexp


def order_spectrum(x, speed, sample_rate, bins):
    speed = np.broadcast_to(np.asarray(speed, float), np.shape(x))
    if np.any(speed <= 0):
        raise ValueError("Shaft speed must be positive in revolutions per second")
    angle = np.cumsum(speed) / sample_rate
    if angle[-1] <= angle[0]:
        raise ValueError("Insufficient angular support")
    uniform = np.linspace(angle[0], angle[-1], 2 * bins)
    signal = np.interp(uniform, angle, x)
    spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(len(signal))))[
        :bins
    ]
    return spectrum / (np.linalg.norm(spectrum) + 1e-12)


def rescale(x, speed, factor):
    speed = np.broadcast_to(np.asarray(speed, float), np.shape(x))
    time = np.arange(max(4, int((len(x) - 1) / factor) + 1)) * factor
    time = time[time <= len(x) - 1]
    return np.interp(time, np.arange(len(x)), x), factor * np.interp(
        time, np.arange(len(x)), speed
    )


def adapter(x, speed, sample_rate, kind, length):
    if kind == "waveform":
        value = np.pad(np.asarray(x)[:length], (0, max(0, length - len(x))))
    elif kind == "spectrum":
        value = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        value = np.interp(
            np.linspace(0, len(value) - 1, length), np.arange(len(value)), value
        )
    elif kind == "order":
        value = order_spectrum(x, speed, sample_rate, length)
    elif kind == "waveform_speed":
        value = np.pad(np.asarray(x)[:length], (0, max(0, length - len(x))))
        return np.stack([value, np.full(length, np.mean(speed))]).astype("float32")
    else:
        raise ValueError(kind)
    return value[None].astype("float32")


def js(p, q):
    mean = (p + q) / 2
    return 0.5 * np.sum(p * np.log((p + 1e-12) / (mean + 1e-12))) + 0.5 * np.sum(
        q * np.log((q + 1e-12) / (mean + 1e-12))
    )


def probe(record, models, c):
    import torch

    data = np.load(record["path"])
    x = data["x"].astype(float)
    speed = data["speed"]
    sample_rate = float(data["sample_rate"])
    reference = order_spectrum(x, speed, sample_rate, c["order_bins"])
    orbit = []
    for factor in c["scales"]:
        xx, ss = rescale(x, speed, factor)
        spectrum = order_spectrum(xx, ss, sample_rate, c["order_bins"])
        if np.linalg.norm(spectrum - reference) <= c["order_tolerance"]:
            orbit.append((xx, ss))
    orbit.insert(0, (x, speed))
    logits = []
    margins = []
    disagreement = []
    for model, settings in models:
        values = []
        for xx, ss in orbit:
            inputs = adapter(
                xx, ss, sample_rate, settings["adapter"], settings["length"]
            )
            with torch.no_grad():
                zz = (
                    model(torch.from_numpy(inputs)[None].to(c["device"]))
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
            values.append(zz)
        posterior = softmax(np.stack(values), axis=-1)
        top = np.sort(posterior, axis=-1)
        logits.append(values[0])
        margins.append(np.mean(top[:, -1] - top[:, -2]))
        disagreement.append(np.mean([js(posterior[0], p) for p in posterior]))
    return np.array(logits), np.array(margins), np.array(disagreement), len(orbit)


def fuse(logits, margin, disagreement, c, calibration, variant):
    normal = lambda x, key: (
        (x - np.array(calibration[key + "_median"]))
        / np.array(calibration[key + "_mad"])
    )
    risk = normal(1 - margin, "margin") + c["risk_weight"] * normal(
        disagreement, "disagreement"
    )
    if variant == "margin":
        risk = normal(1 - margin, "margin")
    if variant == "entropy":
        risk = -np.sum(
            softmax(logits, axis=-1) * np.log(softmax(logits, axis=-1) + 1e-12), axis=-1
        )
    weights = (
        np.ones(len(logits)) / len(logits)
        if variant == "uniform"
        else softmax(-c["temperature"] * risk)
    )
    posterior = (weights[:, None] * softmax(logits, axis=-1)).sum(0)
    energy = -logsumexp((weights[:, None] * logits).sum(0))
    rejection = c["entropy_weight"] * (
        -np.sum(posterior * np.log(posterior + 1e-12))
    ) + (1 - c["entropy_weight"]) * max(energy - calibration["energy_center"], 0.0)
    return posterior, rejection, risk, energy
