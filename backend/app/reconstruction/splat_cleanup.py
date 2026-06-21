"""Post-training cleanup for 3DGS `.splat` assets.

Real-world 3DGS reconstructions (especially of unbounded outdoor scenes) leave
behind a small fraction of junk gaussians that disproportionately wreck the
viewer experience:

  * **Far floaters** — a handful of gaussians flung far from the scene. They're
    reconstruction noise, but the viewer's bounding box is the raw min/max of
    *all* centers, so even 0.1% of points at ±120 units balloon the box, skew
    the auto-framing, and (worse) loom as big translucent planes when you fly
    through them toward the real content.
  * **Faint gaussians** — near-zero opacity; they contribute nothing but cost
    fill-rate and sort time.
  * **Large translucent "haze"** — big, low-opacity splats that fog the scene.

This trims those (a few percent of gaussians) without touching the real
surface, which both cleans up the render and shrinks the asset for the web
viewer. Operates directly on the antimatter15 32-byte format in NumPy.

All thresholds are env-overridable; defaults are deliberately conservative
(they only catch clear outliers) and the function never drops more than
``GSW_CLEAN_MAX_FRACTION`` of the cloud as a safety rail.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_STRIDE = 32

# Opacity (0-255) below this is effectively invisible -> drop.
_FAINT_ALPHA = float(os.environ.get("GSW_CLEAN_FAINT_ALPHA", "6"))
# Large + translucent = haze. Drop if maxscale > _HAZE_SCALE and alpha < _HAZE_ALPHA.
_HAZE_SCALE = float(os.environ.get("GSW_CLEAN_HAZE_SCALE", "0.3"))
_HAZE_ALPHA = float(os.environ.get("GSW_CLEAN_HAZE_ALPHA", "70"))
# Anything this large is junk regardless of opacity.
_HUGE_SCALE = float(os.environ.get("GSW_CLEAN_HUGE_SCALE", "1.0"))
# "Needle" gaussians: one axis vastly longer than another. Healthy disk-splats
# sit around 20:1; anything past this is a spike that renders as a hard streak
# (from sparse poses or a corrupted rotation). Drop them.
_NEEDLE_RATIO = float(os.environ.get("GSW_CLEAN_NEEDLE_RATIO", "100"))
# Far-floater radius = max(_FAR_MIN_RADIUS, percentile(d, _FAR_PCT) * _FAR_FACTOR).
# Use a moderate percentile (98) so the cutoff tracks the real scene's dense
# core rather than the floater tail itself — a 99.5-percentile base is inflated
# by the very floaters we want to drop, leaving junk far from the content.
_FAR_PCT = float(os.environ.get("GSW_CLEAN_FAR_PCT", "98.0"))
_FAR_FACTOR = float(os.environ.get("GSW_CLEAN_FAR_FACTOR", "2.0"))
_FAR_MIN_RADIUS = float(os.environ.get("GSW_CLEAN_FAR_MIN_RADIUS", "8.0"))
# Never drop more than this fraction of the cloud (safety rail against a bad
# scene where the heuristics would gut real geometry).
_MAX_FRACTION = float(os.environ.get("GSW_CLEAN_MAX_FRACTION", "0.25"))

_STRICT = {
    "faint_alpha": float(os.environ.get("GSW_STRICT_CLEAN_FAINT_ALPHA", "18")),
    "haze_scale": float(os.environ.get("GSW_STRICT_CLEAN_HAZE_SCALE", "0.22")),
    "haze_alpha": float(os.environ.get("GSW_STRICT_CLEAN_HAZE_ALPHA", "90")),
    "huge_scale": float(os.environ.get("GSW_STRICT_CLEAN_HUGE_SCALE", "0.7")),
    "needle_ratio": float(os.environ.get("GSW_STRICT_CLEAN_NEEDLE_RATIO", "60")),
    "far_pct": float(os.environ.get("GSW_STRICT_CLEAN_FAR_PCT", "95.0")),
    "far_factor": float(os.environ.get("GSW_STRICT_CLEAN_FAR_FACTOR", "2.0")),
    "far_min_radius": float(os.environ.get("GSW_STRICT_CLEAN_FAR_MIN_RADIUS", "8.0")),
    "max_fraction": float(os.environ.get("GSW_STRICT_CLEAN_MAX_FRACTION", "0.65")),
}


def clean_splat(path: Path, out: Optional[Path] = None, strict: bool = False) -> Tuple[int, int]:
    """Trim floater/haze/faint gaussians from a `.splat` file in place.

    Returns ``(n_before, n_after)``. If the file is tiny or cleanup would drop
    more than the safety fraction, the file is left unchanged.
    """
    path = Path(path)
    out = Path(out) if out is not None else path
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // _STRIDE
    if n < 100:
        return n, n  # too small to bother / not a real reconstruction

    a = raw[: n * _STRIDE].reshape(n, _STRIDE)
    pos = a[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scale = a[:, 12:24].copy().view(np.float32).reshape(n, 3)
    alpha = a[:, 27].astype(np.float32)
    if strict:
        faint_alpha = _STRICT["faint_alpha"]
        haze_scale = _STRICT["haze_scale"]
        haze_alpha = _STRICT["haze_alpha"]
        huge_scale = _STRICT["huge_scale"]
        needle_ratio = _STRICT["needle_ratio"]
        far_pct = _STRICT["far_pct"]
        far_factor = _STRICT["far_factor"]
        far_min_radius = _STRICT["far_min_radius"]
        max_fraction = _STRICT["max_fraction"]
    else:
        faint_alpha = _FAINT_ALPHA
        haze_scale = _HAZE_SCALE
        haze_alpha = _HAZE_ALPHA
        huge_scale = _HUGE_SCALE
        needle_ratio = _NEEDLE_RATIO
        far_pct = _FAR_PCT
        far_factor = _FAR_FACTOR
        far_min_radius = _FAR_MIN_RADIUS
        max_fraction = _MAX_FRACTION

    maxscale = scale.max(axis=1)
    minscale = np.maximum(scale.min(axis=1), 1e-9)
    needle = maxscale / minscale

    center = np.median(pos, axis=0)
    dist = np.linalg.norm(pos - center, axis=1)
    far_radius = max(far_min_radius, float(np.percentile(dist, far_pct)) * far_factor)

    drop = (
        (alpha < faint_alpha)
        | (maxscale > huge_scale)
        | ((maxscale > haze_scale) & (alpha < haze_alpha))
        | (dist > far_radius)
        | (needle > needle_ratio)
    )
    n_drop = int(drop.sum())
    if n_drop == 0:
        if out != path:
            out.write_bytes(raw.tobytes())
        return n, n
    if n_drop > max_fraction * n:
        # Heuristics would gut the cloud — leave it alone, the scene is unusual.
        if out != path:
            out.write_bytes(raw.tobytes())
        return n, n

    kept = a[~drop]
    out.write_bytes(kept.tobytes())
    return n, n - n_drop
