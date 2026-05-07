"""Cohen's kappa calibration harness for the LLM-as-judge scorer.

Usage
-----
Run via the CLI entry-point::

    atlas eval calibrate --labels evals/calibration.csv --out evals/CALIBRATION.md

Or call the functions directly for testing / embedding in other scripts.

The kappa value computed from the bundled stub ``evals/calibration.csv``
is MEANINGLESS — those rows contain placeholder human labels.  Replace
them with real human annotations before trusting the kappa output.  See
``evals/CALIBRATION.md`` (written after each run) for the interpretation.

Cohen's kappa formula
---------------------
Given N judgement pairs (human_score, judge_score) where each score is
binary (0 or 1):

    p_o  = observed agreement fraction = (TP + TN) / N
    p_e  = expected agreement by chance
         = p_human_1 * p_judge_1 + p_human_0 * p_judge_0
    kappa = (p_o - p_e) / (1 - p_e)

where p_human_1 = fraction of human labels == 1, etc.

Kappa interpretation (Landis & Koch 1977):
    < 0       : less than chance agreement
    0.00-0.20 : slight
    0.21-0.40 : fair
    0.41-0.60 : moderate
    0.61-0.80 : substantial
    0.81-1.00 : almost perfect
"""

from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from codex_atlas.judge.factory import make_judge

logger = logging.getLogger(__name__)

_CALIBRATION_CSV_DEFAULT = Path("evals/calibration.csv")
_CALIBRATION_MD_DEFAULT = Path("evals/CALIBRATION.md")


@dataclass
class JudgeAgreement:
    """One labelled sample compared against the judge.

    Attributes:
        question_id: Identifier for the sample (matches the CSV ``qid`` column).
        human_score: Human annotator's binary label (0 or 1).
        judge_score: Judge's binarised score (threshold 0.5).
    """

    question_id: str
    human_score: int  # 0 or 1
    judge_score: int  # 0 or 1


def cohens_kappa(agreements: list[JudgeAgreement]) -> float:
    """Compute Cohen's kappa for a list of judge agreement pairs.

    Args:
        agreements: List of :class:`JudgeAgreement` pairs.

    Returns:
        Cohen's kappa as a float in [-1.0, 1.0].
        Returns 0.0 when the list is empty.
        Returns 0.0 when expected agreement is 1.0 (all labels identical;
        kappa is technically undefined — we return 0.0 to avoid division
        by zero rather than raising, since a degenerate label distribution
        is a data problem, not a code bug).

    Examples::

        # Perfect agreement on balanced labels → kappa = 1.0
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=1),
            JudgeAgreement("q2", human_score=0, judge_score=0),
        ]
        assert cohens_kappa(pairs) == 1.0

        # Perfect disagreement on balanced labels → kappa = -1.0
        pairs = [
            JudgeAgreement("q1", human_score=1, judge_score=0),
            JudgeAgreement("q2", human_score=0, judge_score=1),
        ]
        assert cohens_kappa(pairs) == -1.0
    """
    n = len(agreements)
    if n == 0:
        return 0.0

    tp = sum(1 for a in agreements if a.human_score == 1 and a.judge_score == 1)
    tn = sum(1 for a in agreements if a.human_score == 0 and a.judge_score == 0)
    fp = sum(1 for a in agreements if a.human_score == 0 and a.judge_score == 1)
    fn = sum(1 for a in agreements if a.human_score == 1 and a.judge_score == 0)

    p_o = (tp + tn) / n  # observed agreement

    p_human_1 = (tp + fn) / n
    p_human_0 = (tn + fp) / n
    p_judge_1 = (tp + fp) / n
    p_judge_0 = (tn + fn) / n

    p_e = p_human_1 * p_judge_1 + p_human_0 * p_judge_0  # expected agreement

    if p_e >= 1.0:
        # Degenerate: all labels are the same class.
        return 0.0

    return (p_o - p_e) / (1.0 - p_e)


def load_calibration_csv(path: Path) -> list[dict[str, str]]:
    """Read the calibration CSV and return a list of row dicts.

    Expected columns: ``qid``, ``question``, ``expected_qualified_names``,
    ``answer``, ``citations``, ``human_score`` (0 or 1).

    Args:
        path: Path to the calibration CSV file.

    Returns:
        List of row dicts with string values.

    Raises:
        FileNotFoundError: *path* does not exist.
        ValueError: A required column is missing.
    """
    required = {"qid", "question", "expected_qualified_names", "answer", "human_score"}
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(
                f"calibration.csv is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            rows.append(dict(row))
    return rows


async def run_calibration(
    csv_path: Path = _CALIBRATION_CSV_DEFAULT,
    out_path: Path = _CALIBRATION_MD_DEFAULT,
    *,
    judge_mode: str = "auto",
) -> float:
    """Score the calibration CSV and write a CALIBRATION.md report.

    Args:
        csv_path: Path to the human-labelled CSV file.
        out_path: Path to write the Markdown report.
        judge_mode: Passed to :func:`~codex_atlas.judge.factory.make_judge`.

    Returns:
        The computed Cohen's kappa value.
    """
    rows = load_calibration_csv(csv_path)
    judge = make_judge(mode=judge_mode)
    agreements: list[JudgeAgreement] = []

    for row in rows:
        names = [n.strip() for n in row["expected_qualified_names"].split(";") if n.strip()]
        citations = [
            c.strip()
            for c in row.get("citations", "").split(";")
            if c.strip()
        ]
        result = await judge.score(
            question=row["question"],
            expected_qualified_names=names,
            answer=row["answer"],
            citations=citations,
        )
        judge_binary = 1 if result.score >= 0.5 else 0
        human_binary = int(row["human_score"])
        agreements.append(
            JudgeAgreement(
                question_id=row["qid"],
                human_score=human_binary,
                judge_score=judge_binary,
            )
        )

    kappa = cohens_kappa(agreements)
    _write_calibration_md(agreements, kappa, out_path, judge_mode=judge_mode)
    return kappa


def _write_calibration_md(
    agreements: list[JudgeAgreement],
    kappa: float,
    out_path: Path,
    *,
    judge_mode: str,
) -> None:
    n = len(agreements)
    tp = sum(1 for a in agreements if a.human_score == 1 and a.judge_score == 1)
    tn = sum(1 for a in agreements if a.human_score == 0 and a.judge_score == 0)
    fp = sum(1 for a in agreements if a.human_score == 0 and a.judge_score == 1)
    fn = sum(1 for a in agreements if a.human_score == 1 and a.judge_score == 0)
    agree = tp + tn
    disagree = fp + fn

    divergence_lines = []
    for a in agreements:
        if a.human_score != a.judge_score:
            divergence_lines.append(
                f"| {a.question_id} | {a.human_score} | {a.judge_score} |"
            )
    divergence_block = (
        "\n".join(divergence_lines) if divergence_lines else "_No divergence cases._"
    )

    lines = [
        "# LLM-as-judge calibration report",
        "",
        "> **WARNING — PLACEHOLDER DATA**  ",
        "> The bundled `evals/calibration.csv` contains stub human labels.  ",
        "> This kappa value is **meaningless** until real human annotations replace  ",
        "> the `human_score` column.  See the calibration workflow in INTERPRETATION.md.",
        "",
        "## Summary",
        "",
        f"- Judge mode: `{judge_mode}`",
        f"- Samples: {n}",
        f"- Agreed: {agree} / {n}",
        f"- Disagreed: {disagree} / {n}",
        f"- **Cohen's kappa: {kappa:.4f}**",
        "",
        "### Agreement matrix",
        "",
        "| | Judge=1 | Judge=0 |",
        "| --- | ---: | ---: |",
        f"| Human=1 | {tp} | {fn} |",
        f"| Human=0 | {fp} | {tn} |",
        "",
        "### Divergence cases",
        "",
        "| qid | human | judge |",
        "| --- | --- | --- |",
        divergence_block,
        "",
        "### Kappa interpretation (Landis & Koch 1977)",
        "",
        "| Range | Label |",
        "| --- | --- |",
        "| < 0 | Less than chance |",
        "| 0.00-0.20 | Slight |",
        "| 0.21-0.40 | Fair |",
        "| 0.41-0.60 | Moderate |",
        "| 0.61-0.80 | Substantial |",
        "| 0.81-1.00 | Almost perfect |",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.info("Wrote calibration report to %s", out_path)


def calibration_cli_main(
    csv_path: Path = _CALIBRATION_CSV_DEFAULT,
    out_path: Path = _CALIBRATION_MD_DEFAULT,
    judge_mode: str = "auto",
) -> None:
    """Sync wrapper for the async ``run_calibration`` — used by the CLI."""
    kappa = asyncio.run(run_calibration(csv_path, out_path, judge_mode=judge_mode))
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Report written to: {out_path}")


__all__ = [
    "JudgeAgreement",
    "calibration_cli_main",
    "cohens_kappa",
    "load_calibration_csv",
    "run_calibration",
]
