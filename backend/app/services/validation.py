"""Heuristic coverage / quality scoring for the upload validation report.

These are deliberately simple, explainable heuristics (no ML): coverage scales
with photo count toward the recommended target, quality is driven by average
sharpness, and we surface friendly warnings (too few photos, blur, "few side
views") matching the UX tone in 04_user_experience.md.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .. import config

# Sharpness (variance-of-Laplacian) below this reads as "soft / blurry".
BLUR_SHARPNESS = 50.0
# Sharpness at/above this is treated as fully sharp for scoring.
SHARP_SATURATION = 400.0


def _coverage_score(photo_count: int) -> float:
    """0..1 — scales with count toward the recommended target."""
    if photo_count <= 0:
        return 0.0
    score = photo_count / config.RECOMMENDED_PHOTOS
    return float(min(1.0, score))


def _quality_score(sharpness_values: List[float]) -> float:
    """0..1 — average sharpness normalized to a saturation point."""
    if not sharpness_values:
        return 0.0
    avg = sum(sharpness_values) / len(sharpness_values)
    return float(min(1.0, avg / SHARP_SATURATION))


def score_uploads(
    photo_count: int, sharpness_values: List[float]
) -> Tuple[float, float, List[str], bool]:
    """Return (coverage_score, quality_score, warnings, ready_to_reconstruct)."""
    warnings: List[str] = []

    coverage = _coverage_score(photo_count)
    quality = _quality_score(sharpness_values)

    ready = config.MIN_PHOTOS <= photo_count <= config.MAX_PHOTOS

    if photo_count == 0:
        warnings.append("No photos yet. Add some to get started.")
    elif photo_count < config.MIN_PHOTOS:
        warnings.append(
            f"We need at least {config.MIN_PHOTOS} photos to build a 3D world. "
            f"Try adding more photos from different angles."
        )
    elif photo_count < config.RECOMMENDED_PHOTOS:
        warnings.append(
            f"For a fuller result, aim for around {config.RECOMMENDED_PHOTOS} photos. "
            f"More angles mean fewer holes."
        )

    if photo_count > config.MAX_PHOTOS:
        warnings.append(
            f"That's a lot of photos! We'll use the first {config.MAX_PHOTOS}; "
            f"extra photos beyond that were skipped."
        )

    # Blur warning based on how many photos are soft.
    if sharpness_values:
        blurry = sum(1 for s in sharpness_values if s < BLUR_SHARPNESS)
        if blurry and blurry >= max(1, len(sharpness_values) // 3):
            warnings.append(
                "Some photos look a little blurry. Sharper photos give a "
                "cleaner 3D result."
            )

    # "Few side views" style coverage hint. Without real pose data we use a
    # count-based proxy: a modest set is likely front-heavy.
    if config.MIN_PHOTOS <= photo_count < config.RECOMMENDED_PHOTOS:
        warnings.append(
            "We may have many front views but few side views. The 3D result "
            "could have holes on the sides — walk around and capture more angles."
        )

    return round(coverage, 3), round(quality, 3), warnings, ready


def build_report(uploads: List[Dict], rejected: List[Dict]) -> Dict:
    photo_count = len(uploads)
    sharpness_values = [u.get("sharpness", 0.0) for u in uploads]
    coverage, quality, warnings, ready = score_uploads(photo_count, sharpness_values)
    return {
        "accepted": photo_count,
        "rejected": rejected,
        "photo_count": photo_count,
        "coverage_score": coverage,
        "quality_score": quality,
        "warnings": warnings,
        "ready_to_reconstruct": ready,
    }
