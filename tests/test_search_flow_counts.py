from __future__ import annotations

import pandas as pd

from clustro.reporting.exports import _build_search_flow_frame


def test_search_flow_counts_repair_stages() -> None:
    registry = pd.DataFrame(
        {
            "candidate_id": ["compat", "pruned", "hard", "top", "accepted"],
            "search_stage": [
                "compatibility_rejected",
                "pilot_pruned",
                "full_evaluated",
                "full_evaluated",
                "full_evaluated",
            ],
            "accepted_before_top_fraction": [False, False, False, True, True],
            "accepted": [False, False, False, False, True],
            "rejection_reasons": [
                "incompatible",
                "silhouette_too_low",
                "cluster_too_small",
                "outside_top_fraction_policy",
                "",
            ],
        }
    )

    counts = dict(
        zip(
            _build_search_flow_frame(registry)["stage"],
            _build_search_flow_frame(registry)["count"],
            strict=True,
        )
    )

    assert counts["generated_total"] == 5
    assert counts["compatibility_rejected"] == 1
    assert counts["pilot_pruned"] == 1
    assert counts["full_evaluated"] == 3
    assert counts["hard_rejected"] == 1
    assert counts["accepted_before_top_fraction"] == 2
    assert counts["accepted_final"] == 1
    assert counts["top_fraction_rejected"] == 1
    assert counts["consensus_used"] == 1
