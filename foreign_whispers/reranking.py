"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    import re

    CHARS_PER_SECOND = 15.0
    budget_chars = int(target_duration_s * CHARS_PER_SECOND)

    # If baseline already fits within budget, no shorter candidates needed.
    if len(baseline_es) <= budget_chars:
        return []

    candidates: list[TranslationCandidate] = []
    seen: set[str] = set()

    def _add(text: str, rationale: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if text and text not in seen and text != baseline_es:
            seen.add(text)
            candidates.append(TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            ))

    # ── Strategy 1: Remove Spanish filler words / redundant phrases ──────────
    FILLER_SUBS = [
        (r"\ben este momento\b", "ahora"),
        (r"\ben este instante\b", "ahora"),
        (r"\bactualmente\b", "hoy"),
        (r"\bpor supuesto(que)?\b", "claro"),
        (r"\bde hecho\b", ""),
        (r"\bsegún parece\b", ""),
        (r"\bpor lo tanto\b", "así"),
        (r"\bsin embargo\b", "pero"),
        (r"\ba continuación\b", ""),
        (r"\bademás\b", ""),
        (r"\bes decir\b", "o sea"),
        (r"\bpor otro lado\b", ""),
        (r"\ben realidad\b", ""),
        (r"\ben general\b", ""),
        (r"\blo que es más\b", ""),
        (r"\bhay que tener en cuenta que\b", ""),
        (r"\bcabe destacar que\b", ""),
        (r"\bse puede decir que\b", ""),
        (r"\bcon respecto a\b", "sobre"),
        (r"\ben lo que respecta a\b", "sobre"),
    ]
    text_filler = baseline_es
    for pattern, replacement in FILLER_SUBS:
        text_filler = re.sub(pattern, replacement, text_filler, flags=re.IGNORECASE)
    text_filler = re.sub(r"\s+", " ", text_filler).strip()
    if len(text_filler) < len(baseline_es):
        _add(text_filler, "Removed filler words and redundant phrases")

    # ── Strategy 2: Abbreviate common long Spanish multi-word expressions ─────
    ABBREVS = [
        (r"\bEstados Unidos\b", "EE.UU."),
        (r"\bNaciones Unidas\b", "la ONU"),
        (r"\bpor ciento\b", "%"),
        (r"\bmillones de dólares\b", "M$"),
        (r"\bmillones\b", "M"),
        (r"\bel gobierno de los\b", "el gobierno"),
        (r"\bla situación de\b", "la situación en"),
        (r"\bel presidente de\b", "el pdte. de"),
        (r"\bsecretary of state\b", "canciller"),   # stray EN phrase
    ]
    text_abbrev = baseline_es
    for pattern, replacement in ABBREVS:
        text_abbrev = re.sub(pattern, replacement, text_abbrev, flags=re.IGNORECASE)
    text_abbrev = re.sub(r"\s+", " ", text_abbrev).strip()
    if len(text_abbrev) < len(baseline_es):
        _add(text_abbrev, "Abbreviated common long phrases")

    # ── Strategy 3: Truncate to budget at a natural word / sentence boundary ──
    if len(baseline_es) > budget_chars:
        truncated = baseline_es[:budget_chars]
        # Prefer sentence-level break
        for sep in (". ", "? ", "! "):
            idx = truncated.rfind(sep)
            if idx > budget_chars // 2:
                truncated = truncated[: idx + 1].strip()
                break
        else:
            # Fall back to last comma or word boundary
            for sep in (", ", " "):
                idx = truncated.rfind(sep)
                if idx > budget_chars // 3:
                    truncated = truncated[:idx].strip()
                    break
        _add(truncated, "Truncated to fit duration budget")

    # Sort shortest-first so caller can pick the closest-fitting candidate.
    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: budget=%d chars, baseline=%d chars → %d candidates",
        budget_chars,
        len(baseline_es),
        len(candidates),
    )
    return candidates
