#!/usr/bin/env python3
"""
Muse LSL -> Godot signal processing bridge (with Focused-High detector).

Adds a robust, stateful "focused_high" signal intended to correspond to:
  focused high ≈ elevated (theta + low-beta) with controlled arousal
               (penalize delta + high-beta dominance; do not require alpha to rise,
                but penalize alpha collapse and reward alpha rebound/trend).

Core ideas (computed on each sliding window):
  1) Band powers P_band via FFT-PSD integration.
  2) Relative band powers r_band = P_band / (P_total + eps) for scale invariance.
  3) Online baseline μ, σ for each r_band via exponential moving statistics.
  4) Z-scores z_band = (r_band - μ_band) / (σ_band + eps).
  5) Focused-high score in [0,1] via logistic transform of a weighted z-feature.
  6) Focus meter (0..1) rises quickly when focused-high score is high, decays otherwise.

Dependencies:
  pip install pylsl numpy
Optional:
  pip install scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

print("running godotlsl")

try:
    from pylsl import StreamInlet
    try:
        from pylsl import resolve_byprop as _resolve_byprop
    except Exception:
        _resolve_byprop = None
    try:
        from pylsl import resolve_stream as _resolve_stream
    except Exception:
        _resolve_stream = None
    try:
        from pylsl import resolve_streams as _resolve_streams
    except Exception:
        _resolve_streams = None
except Exception as exc:  # pragma: no cover
    print("ERROR: pylsl is required. Install with `pip install pylsl`.")
    raise

try:
    from scipy.signal import butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    HAVE_TK = True
except Exception:
    HAVE_TK = False


# Canonical bands + splits for better "focused high" discrimination.
# High beta is more often "stressy arousal"; low beta is more "task engagement".
BANDS_HZ = {
    "delta":   (1.0, 4.0),
    "theta":   (4.0, 8.0),
    "alpha":   (8.0, 13.0),
    "beta":    (13.0, 30.0),
    "beta_lo": (13.0, 20.0),
    "beta_hi": (20.0, 30.0),
    "gamma":   (30.0, 45.0),
}

TOTAL_BAND = (1.0, 45.0)  # used for relative power


@dataclass
class Config:
    stream_name_contains: str
    stream_type: str
    max_resolve_seconds: float
    udp_host: str
    udp_port: int
    window_seconds: float
    step_seconds: float
    lsl_timeout: float
    detrend: bool
    use_filter: bool
    verbose: bool
    # Focused-high parameters
    baseline_seconds: float
    baseline_alpha: float
    adapt_alpha: float
    sigma_floor: float
    score_k: float
    score_bias: float
    w_theta: float
    w_beta_lo: float
    w_beta_hi: float
    w_delta: float
    w_alpha: float
    w_alpha_trend: float
    clip_z: float
    meter_rise: float
    meter_decay: float
    alpha_collapse_z: float
    max_artifact_rel_total: float


class UDPJsonSender:
    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: Dict) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.addr)


class TextWindow:
    def __init__(self, title: str = "Muse Band Power") -> None:
        if not HAVE_TK:
            raise RuntimeError("tkinter is not available in this Python build.")
        self.root = tk.Tk()
        self.root.title(title)
        self.text = ScrolledText(self.root, width=80, height=22, state="disabled")
        self.text.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def append_line(self, line: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")
        self.root.update()


class SlidingWindow:
    def __init__(self, maxlen: int, n_channels: int) -> None:
        self.n_channels = n_channels
        self.buffers: List[Deque[float]] = [deque(maxlen=maxlen) for _ in range(n_channels)]
        self.timestamps: Deque[float] = deque(maxlen=maxlen)

    def push(self, sample: List[float], ts: float) -> None:
        for i in range(self.n_channels):
            self.buffers[i].append(sample[i])
        self.timestamps.append(ts)

    def is_full(self) -> bool:
        return len(self.timestamps) == self.timestamps.maxlen

    def to_array(self) -> np.ndarray:
        return np.vstack([np.array(buf) for buf in self.buffers])

    def last_timestamp(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


def _resolve_by_type(stream_type: str, timeout: float):
    if _resolve_byprop is not None:
        return _resolve_byprop("type", stream_type, timeout=timeout)
    if _resolve_stream is not None:
        return _resolve_stream("type", stream_type, timeout=timeout)
    if _resolve_streams is not None:
        return _resolve_streams(timeout=timeout)
    raise RuntimeError("No LSL resolve function available in pylsl.")


def resolve_lsl_stream(cfg: Config):
    start = time.time()
    while time.time() - start < cfg.max_resolve_seconds:
        streams = _resolve_by_type(cfg.stream_type, timeout=1.0)
        for s in streams:
            if s.type().lower() != cfg.stream_type.lower():
                continue
            if cfg.stream_name_contains.lower() in s.name().lower():
                return s
    return None


def bandpower_fft(signal: np.ndarray, sf: float, band: Tuple[float, float]) -> float:
    n = len(signal)
    if n == 0 or sf <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sf)
    fft = np.fft.rfft(signal * np.hanning(n))
    psd = (np.abs(fft) ** 2) / max(n, 1)
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx):
        return 0.0
    # np.trapz is broadly compatible across numpy versions.
    return float(np.trapz(psd[idx], freqs[idx]))


def bandpass_filter(data: np.ndarray, sf: float, band: Tuple[float, float]) -> np.ndarray:
    if not HAVE_SCIPY or sf <= 0:
        return data
    low, high = band
    nyq = 0.5 * sf
    lowc = max(low / nyq, 1e-6)
    highc = min(high / nyq, 0.99)
    if highc <= lowc:
        return data
    b, a = butter(4, [lowc, highc], btype="band")
    return filtfilt(b, a, data)


def estimate_sample_rate(timestamps: Deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0.0:
        return 0.0
    return (len(timestamps) - 1) / duration


def compute_band_features(window: np.ndarray, sf: float, cfg: Config) -> Dict[str, List[float]]:
    # window shape: channels x samples
    if cfg.detrend:
        window = window - np.mean(window, axis=1, keepdims=True)

    out: Dict[str, List[float]] = {}
    for band_name, band in BANDS_HZ.items():
        vals: List[float] = []
        for ch in range(window.shape[0]):
            sig = window[ch]
            if cfg.use_filter:
                sig = bandpass_filter(sig, sf, band)
            vals.append(bandpower_fft(sig, sf, band))
        out[band_name] = vals

    return out


def mean_across_channels(features: Dict[str, List[float]]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for k, vals in features.items():
        if not vals:
            m[k] = 0.0
        else:
            m[k] = float(np.mean(vals))
    return m


def rel_powers(mean_powers: Dict[str, float], eps: float = 1e-12) -> Dict[str, float]:
    # Relative to total 1..45 Hz power to reduce scaling differences (electrode contact, gain, etc.)
    p_total = 0.0
    for name, p in mean_powers.items():
        # Use only bands inside TOTAL_BAND. All of ours are.
        p_total += max(p, 0.0)

    denom = p_total + eps
    r: Dict[str, float] = {}
    for name, p in mean_powers.items():
        r[name] = max(p, 0.0) / denom
    r["_total"] = p_total
    return r


def clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    # Stable logistic
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class OnlineBaseline:
    """
    Exponential moving mean/variance for each key.

    Update rules for each feature x:
      μ <- (1-a) μ + a x
      v <- (1-a) v + a (x-μ)^2
      σ <- sqrt(v)

    Two phases:
      - baseline phase: faster alpha (cfg.baseline_alpha) for first baseline_seconds
      - adapt phase: slower alpha (cfg.adapt_alpha) thereafter
    """

    def __init__(self, keys: List[str], sigma_floor: float, baseline_alpha: float, adapt_alpha: float) -> None:
        self.keys = keys
        self.mu = {k: 0.0 for k in keys}
        self.var = {k: 1.0 for k in keys}  # start nonzero
        self.sigma_floor = sigma_floor
        self.baseline_alpha = baseline_alpha
        self.adapt_alpha = adapt_alpha
        self.samples = 0

    def update(self, x: Dict[str, float], alpha: float) -> None:
        a = clip(alpha, 1e-6, 1.0)
        for k in self.keys:
            xv = float(x.get(k, 0.0))
            mu = self.mu[k]
            mu_new = (1.0 - a) * mu + a * xv
            # use new mean for variance update (reduces bias a bit)
            dv = xv - mu_new
            var_new = (1.0 - a) * self.var[k] + a * (dv * dv)
            self.mu[k] = mu_new
            self.var[k] = max(var_new, self.sigma_floor * self.sigma_floor)
        self.samples += 1

    def zscores(self, x: Dict[str, float]) -> Dict[str, float]:
        z: Dict[str, float] = {}
        for k in self.keys:
            xv = float(x.get(k, 0.0))
            sigma = math.sqrt(max(self.var[k], self.sigma_floor * self.sigma_floor))
            z[k] = (xv - self.mu[k]) / sigma
        return z


class FocusedHighDetector:
    """
    Produces:
      - focused_high_score in [0,1]
      - focused_high_meter in [0,1] (leaky integrator for gameplay/UI stability)

    Feature construction (working model):
      Let z_* be z-score of relative power for each band.

      Focus feature F:
        F =  w_beta_lo * clip(z_beta_lo)
           + w_theta   * clip(z_theta)
           + w_alpha   * clip(z_alpha)
           + w_alpha_trend * clip(z_alpha_trend)
           - w_beta_hi * relu(clip(z_beta_hi))
           - w_delta   * relu(clip(z_delta))

      Score:
        score = sigmoid(score_k * (F - score_bias))

    Safety/robustness:
      - alpha collapse penalty by gating score down when z_alpha << 0.
      - artifact gating: if total power too large relative to baseline, damp score.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.meter = 0.0
        self.alpha_ema = 0.0
        self.prev_alpha_ema = 0.0
        self.total_ema = 0.0

    def step(self, rel: Dict[str, float], z: Dict[str, float], dt: float) -> Dict[str, float]:
        cfg = self.cfg

        # Smooth alpha to extract a trend term (rebound proxy).
        self.prev_alpha_ema = self.alpha_ema
        self.alpha_ema = 0.8 * self.alpha_ema + 0.2 * float(rel.get("alpha", 0.0))
        alpha_trend = (self.alpha_ema - self.prev_alpha_ema) / max(dt, 1e-6)

        # Smooth total power (artifact proxy). Large abrupt increases often come from motion/jaw/EMG.
        self.total_ema = 0.9 * self.total_ema + 0.1 * float(rel.get("_total", 0.0))

        def cz(name: str) -> float:
            return clip(float(z.get(name, 0.0)), -cfg.clip_z, cfg.clip_z)

        def relu(x: float) -> float:
            return x if x > 0.0 else 0.0

        z_theta   = cz("theta")
        z_beta_lo = cz("beta_lo")
        z_beta_hi = cz("beta_hi")
        z_delta   = cz("delta")
        z_alpha   = cz("alpha")

        # Trend z-score surrogate: normalize by a rough scale to keep it comparable.
        # dt is ~step_seconds; alpha_trend is small; scale factor is empirical.
        z_alpha_trend = clip(alpha_trend * 50.0, -cfg.clip_z, cfg.clip_z)

        # Weighted feature score.
        F = (
            cfg.w_beta_lo * z_beta_lo +
            cfg.w_theta   * z_theta +
            cfg.w_alpha   * z_alpha +
            cfg.w_alpha_trend * z_alpha_trend -
            cfg.w_beta_hi * relu(z_beta_hi) -
            cfg.w_delta   * relu(z_delta)
        )

        score = sigmoid(cfg.score_k * (F - cfg.score_bias))

        # Gate down if alpha is collapsing (often stress/over-arousal, eye/face artifacts).
        # If z_alpha is very negative, cap score.
        if z_alpha < cfg.alpha_collapse_z:
            # Linear cap from collapse point down to stronger collapse.
            # Example: alpha_collapse_z=-1.25 => below that begins to suppress.
            severity = clip((cfg.alpha_collapse_z - z_alpha) / 2.0, 0.0, 1.0)
            score *= (1.0 - 0.7 * severity)

        # Artifact damping: if total power explodes relative to its own EMA, damp score.
        # Uses relative ratio; protects beta/gamma contamination.
        total = float(rel.get("_total", 0.0))
        denom = max(self.total_ema, 1e-9)
        ratio = total / denom
        if ratio > cfg.max_artifact_rel_total:
            # Hard damp when obviously noisy.
            score *= 0.2

        # Leaky integrator meter (stable “value that rises”).
        # m <- m + rise*score*dt - decay*(1-score)*dt
        self.meter = clip(
            self.meter + cfg.meter_rise * score * dt - cfg.meter_decay * (1.0 - score) * dt,
            0.0,
            1.0,
        )

        return {
            "F": float(F),
            "score": float(score),
            "meter": float(self.meter),
            "z_theta": float(z_theta),
            "z_beta_lo": float(z_beta_lo),
            "z_beta_hi": float(z_beta_hi),
            "z_delta": float(z_delta),
            "z_alpha": float(z_alpha),
            "z_alpha_trend": float(z_alpha_trend),
            "artifact_ratio": float(ratio),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Muse LSL to Godot signal processing bridge")
    parser.add_argument("--name-contains", default="Muse", help="LSL stream name substring")
    parser.add_argument("--type", default="EEG", dest="stream_type", help="LSL stream type")
    parser.add_argument("--resolve-seconds", type=float, default=15.0, help="time to resolve LSL stream")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Godot UDP host")
    parser.add_argument("--udp-port", type=int, default=12000, help="Godot UDP port")
    parser.add_argument("--window", type=float, default=2.0, help="window size in seconds")
    parser.add_argument("--step", type=float, default=0.25, help="step size in seconds")
    parser.add_argument("--lsl-timeout", type=float, default=1.0, help="LSL pull timeout")
    parser.add_argument("--detrend", action="store_true", help="remove mean from window")
    parser.add_argument("--filter", action="store_true", help="apply bandpass filter before FFT")
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--text", action="store_true", help="print values periodically")
    parser.add_argument("--text-interval", type=float, default=5.0, help="seconds between text prints")
    parser.add_argument("--text-window", action="store_true", help="show a simple text window")
    parser.add_argument("--json", action="store_true", help="enable JSON file output")
    parser.add_argument("--json-file", default="muse_latest.json", help="path for JSON output file")
    parser.add_argument("--json-interval", type=float, default=5.0, help="seconds between JSON file writes")
    parser.add_argument("--raw-send", action="store_true", help="send individual feature values as plain text to 127.0.0.1:5005")

    # Focused-high tuning knobs
    parser.add_argument("--baseline-seconds", type=float, default=30.0, help="seconds used to build baseline statistics")
    parser.add_argument("--baseline-alpha", type=float, default=0.08, help="EMA alpha during baseline build")
    parser.add_argument("--adapt-alpha", type=float, default=0.005, help="EMA alpha after baseline (slow drift adaptation)")
    parser.add_argument("--sigma-floor", type=float, default=1e-4, help="minimum std-dev for z-score stability")

    parser.add_argument("--score-k", type=float, default=1.6, help="logistic slope for focused-high score")
    parser.add_argument("--score-bias", type=float, default=0.6, help="bias/threshold for focused-high feature F")

    parser.add_argument("--w-theta", type=float, default=0.9, help="weight for theta z-score")
    parser.add_argument("--w-beta-lo", type=float, default=1.2, help="weight for low-beta z-score")
    parser.add_argument("--w-beta-hi", type=float, default=0.9, help="penalty weight for high-beta z-score (relu)")
    parser.add_argument("--w-delta", type=float, default=0.8, help="penalty weight for delta z-score (relu)")
    parser.add_argument("--w-alpha", type=float, default=0.2, help="weak reward for alpha z-score")
    parser.add_argument("--w-alpha-trend", type=float, default=0.3, help="reward for alpha rebound/trend")

    parser.add_argument("--clip-z", type=float, default=3.0, help="clip z-scores to [-clip-z, clip-z]")
    parser.add_argument("--meter-rise", type=float, default=0.9, help="meter rise rate per second")
    parser.add_argument("--meter-decay", type=float, default=0.45, help="meter decay rate per second")
    parser.add_argument("--alpha-collapse-z", type=float, default=-1.25, help="begin suppressing score when z_alpha below this")
    parser.add_argument("--max-artifact-rel-total", type=float, default=2.2, help="damp score if total power / EMA(total) exceeds this")
    args = parser.parse_args()

    cfg = Config(
        stream_name_contains=args.name_contains,
        stream_type=args.stream_type,
        max_resolve_seconds=args.resolve_seconds,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        window_seconds=args.window,
        step_seconds=args.step,
        lsl_timeout=args.lsl_timeout,
        detrend=args.detrend,
        use_filter=args.filter,
        verbose=args.verbose,
        baseline_seconds=args.baseline_seconds,
        baseline_alpha=args.baseline_alpha,
        adapt_alpha=args.adapt_alpha,
        sigma_floor=args.sigma_floor,
        score_k=args.score_k,
        score_bias=args.score_bias,
        w_theta=args.w_theta,
        w_beta_lo=args.w_beta_lo,
        w_beta_hi=args.w_beta_hi,
        w_delta=args.w_delta,
        w_alpha=args.w_alpha,
        w_alpha_trend=args.w_alpha_trend,
        clip_z=args.clip_z,
        meter_rise=args.meter_rise,
        meter_decay=args.meter_decay,
        alpha_collapse_z=args.alpha_collapse_z,
        max_artifact_rel_total=args.max_artifact_rel_total,
    )

    stream = resolve_lsl_stream(cfg)
    if stream is None:
        print("ERROR: No LSL stream found. Is Muse streaming to LSL?")
        return 1

    inlet = StreamInlet(stream)
    info = inlet.info()
    n_channels = info.channel_count()
    nominal_srate = float(info.nominal_srate())

    if cfg.verbose:
        print(f"Connected to LSL stream: {info.name()} ({n_channels} ch @ {nominal_srate} Hz)")

    samples_per_window = max(8, int(round(cfg.window_seconds * nominal_srate)))
    samples_per_step = max(1, int(round(cfg.step_seconds * nominal_srate)))

    window = SlidingWindow(samples_per_window, n_channels)
    sender = UDPJsonSender(cfg.udp_host, cfg.udp_port)
    text_window = TextWindow("Muse Band Power + Focused High") if args.text_window else None

    last_text_ts = 0.0
    last_json_ts = 0.0

    samples_since_last = 0
    last_send_ts = 0.0

    # Baseline keys: use relative powers for stability.
    baseline_keys = ["delta", "theta", "alpha", "beta_lo", "beta_hi"]
    baseline = OnlineBaseline(
        keys=baseline_keys,
        sigma_floor=cfg.sigma_floor,
        baseline_alpha=cfg.baseline_alpha,
        adapt_alpha=cfg.adapt_alpha,
    )
    detector = FocusedHighDetector(cfg)

    t0 = time.time()
    baseline_done = False

    while True:
        sample, ts = inlet.pull_sample(timeout=cfg.lsl_timeout)
        if sample is None:
            continue

        window.push(sample, ts)
        samples_since_last += 1

        if not window.is_full():
            continue

        if samples_since_last < samples_per_step:
            continue

        samples_since_last = 0

        sf = estimate_sample_rate(window.timestamps) or nominal_srate
        data = window.to_array()

        # 1) Per-channel band powers (for debugging/visualization)
        band_powers_ch = compute_band_features(data, sf, cfg)

        # 2) Mean across channels (more stable on Muse than per-electrode decisions)
        mean_powers = mean_across_channels(band_powers_ch)

        # 3) Relative powers
        rel = rel_powers(mean_powers)

        # 4) Update baseline statistics for z-scoring
        elapsed = time.time() - t0
        if elapsed < cfg.baseline_seconds:
            baseline.update(rel, alpha=cfg.baseline_alpha)
        else:
            baseline_done = True
            baseline.update(rel, alpha=cfg.adapt_alpha)

        z = baseline.zscores(rel)

        # 5) Compute focused-high score + meter
        focused = detector.step(rel, z, dt=cfg.step_seconds)

        payload = {
            "t": window.last_timestamp(),
            "sf": sf,
            "channels": n_channels,
            "baseline_ready": bool(baseline_done),
            # Raw per-channel band powers (original behavior; includes added beta_lo/beta_hi).
            "features": band_powers_ch,
            # Collapsed/normalized features used by detector.
            "rel": {k: float(rel.get(k, 0.0)) for k in ["delta", "theta", "alpha", "beta", "beta_lo", "beta_hi", "gamma"]} | {"_total": float(rel.get("_total", 0.0))},
            "z": {k: float(z.get(k, 0.0)) for k in baseline_keys},
            # Main deliverable: value that rises during focused-high.
            "focused_high": focused,
        }

        if args.raw_send:
            for band_values in band_powers_ch.values():
                for v in band_values:
                    sender.sock.sendto(f"{v}".encode(), ("127.0.0.1", 5005))
            last_send_ts = time.time()
        else:
            sender.send(payload)
            last_send_ts = time.time()

        if args.json:
            now = time.time()
            if now - last_json_ts >= args.json_interval:
                tmp_dir = os.path.dirname(args.json_file) or "."
                with tempfile.NamedTemporaryFile("w", dir=tmp_dir, delete=False) as tmp:
                    json.dump(payload, tmp, separators=(",", ":"))
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_name = tmp.name
                os.replace(tmp_name, args.json_file)
                last_json_ts = now

        if args.text or text_window is not None:
            now = time.time()
            if now - last_text_ts >= args.text_interval:
                # Display short status: band means + focused high score/meter
                parts = []
                for band in ["delta", "theta", "alpha", "beta_lo", "beta_hi"]:
                    parts.append(f"{band}_rel={rel.get(band,0.0):.3f} z={z.get(band,0.0):+.2f}")
                parts.append(f"F={focused['F']:+.2f} score={focused['score']:.3f} meter={focused['meter']:.3f} art={focused['artifact_ratio']:.2f}")
                if not baseline_done:
                    parts.append(f"(baseline {elapsed:.1f}/{cfg.baseline_seconds:.0f}s)")
                line = " | ".join(parts)
                if args.text:
                    print(line)
                    sys.stdout.flush()
                if text_window is not None:
                    text_window.append_line(line)
                last_text_ts = now

        if cfg.verbose:
            print(f"sent @ {last_send_ts:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
