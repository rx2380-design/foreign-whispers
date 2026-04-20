"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Scores four independent quality dimensions, each normalised to [0, 1]
    (higher = better), plus a weighted ``overall`` composite.

    Dimensions
    ----------
    timing_accuracy
        How well TTS durations fit the source time windows.
        Based on mean absolute duration error capped at 3s and the
        percentage of segments with severe stretch (>1.4x).

    drift_penalty
        How much the dubbed timeline has shifted relative to the original.
        Total cumulative drift of up to 10s is penalised linearly.

    naturalness
        Speaking rate consistency across segments.
        High variance in predicted TTS rates indicates uneven pacing.
        Measured as the coefficient of variation of per-segment chars/s.

    coverage
        Fraction of segments that are successfully dubbed (i.e. not FAIL).
        FAIL segments fall back to silence — audible gaps to the viewer.

    overall
        Weighted average: timing 40%, drift 20%, naturalness 20%, coverage 20%.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        aligned: Scheduled segments from ``global_align`` or ``global_align_dp``.
        align_report: Optional pre-computed ``clip_evaluation_report`` dict.
            If *None*, it is computed from *metrics* and *aligned*.

    Returns:
        Dict with keys ``timing_accuracy``, ``drift_penalty``, ``naturalness``,
        ``coverage``, ``overall``, and a ``raw`` sub-dict of the underlying
        numbers for transparency.
    """
    if not metrics:
        zero = {"timing_accuracy": 0.0, "drift_penalty": 0.0,
                "naturalness": 0.0, "coverage": 0.0, "overall": 0.0, "raw": {}}
        return zero

    report = align_report or clip_evaluation_report(metrics, aligned)

    # ── Timing accuracy ──────────────────────────────────────────────────────
    # Mean abs error: 0s → 1.0, ≥3s → 0.0  (linear)
    mae = report["mean_abs_duration_error_s"]
    timing_from_mae = max(0.0, 1.0 - mae / 3.0)

    # Severe stretch pct: 0% → 1.0, ≥50% → 0.0
    pct_severe = report["pct_severe_stretch"]
    timing_from_stretch = max(0.0, 1.0 - pct_severe / 50.0)

    timing_accuracy = round((timing_from_mae + timing_from_stretch) / 2, 3)

    # ── Drift penalty ────────────────────────────────────────────────────────
    # Drift 0s → 1.0, ≥10s → 0.0
    drift = abs(report["total_cumulative_drift_s"])
    drift_penalty = round(max(0.0, 1.0 - drift / 10.0), 3)

    # ── Naturalness (speaking rate consistency) ──────────────────────────────
    # Rate = tgt_char_count / source_duration_s per segment
    rates = [
        m.tgt_char_count / m.source_duration_s
        for m in metrics
        if m.source_duration_s > 0
    ]
    if len(rates) >= 2:
        mean_rate = _stats.mean(rates)
        stdev_rate = _stats.stdev(rates)
        cv = stdev_rate / mean_rate if mean_rate > 0 else 1.0
        # CV of 0 → 1.0 (perfectly consistent), CV ≥ 1.0 → 0.0
        naturalness = round(max(0.0, 1.0 - cv), 3)
    else:
        naturalness = 1.0

    # ── Coverage ─────────────────────────────────────────────────────────────
    n_fail = sum(1 for a in aligned if a.action == AlignAction.FAIL)
    coverage = round(1.0 - n_fail / max(len(aligned), 1), 3)

    # ── Overall ──────────────────────────────────────────────────────────────
    overall = round(
        0.40 * timing_accuracy
        + 0.20 * drift_penalty
        + 0.20 * naturalness
        + 0.20 * coverage,
        3,
    )

    return {
        "timing_accuracy": timing_accuracy,
        "drift_penalty":   drift_penalty,
        "naturalness":     naturalness,
        "coverage":        coverage,
        "overall":         overall,
        "raw": {
            "mean_abs_duration_error_s": mae,
            "pct_severe_stretch":        pct_severe,
            "total_cumulative_drift_s":  report["total_cumulative_drift_s"],
            "n_fail":                    n_fail,
            "n_segments":                len(metrics),
        },
    }
