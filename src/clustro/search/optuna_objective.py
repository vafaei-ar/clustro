"""Optional Optuna integration for candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import optuna

from clustro.search.search_space import Candidate


@dataclass(slots=True)
class TrialBundle:
    trial: optuna.trial.Trial
    candidate: Candidate


def create_study(name: str) -> optuna.Study:
    return optuna.create_study(direction="maximize", study_name=name)
