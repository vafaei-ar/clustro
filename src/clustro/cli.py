"""Command-line interface for clustro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clustro import Experiment


def app() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate-config":
        experiment = Experiment.from_yaml(args.config)
        print(json.dumps(experiment.validate().model_dump(mode="json"), indent=2, default=str))
        return

    if args.command == "inspect-data":
        experiment = Experiment.from_yaml(args.config)
        print(json.dumps(experiment.inspect_data(), indent=2))
        return

    if args.command == "run":
        Experiment.from_yaml(args.config).run()
        print(f"Run complete: {Path(args.config).expanduser().resolve()}")
        return

    if args.command == "resume":
        Experiment.from_output_dir(args.experiment_id).resume()
        print(f"Resume complete: {args.experiment_id}")
        return

    if args.command == "status":
        status = Experiment.from_output_dir(args.experiment_id).status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.command == "consensus":
        Experiment.from_output_dir(args.experiment_id).build_consensus()
        print(f"Consensus complete: {args.experiment_id}")
        return

    if args.command == "report":
        Experiment.from_output_dir(args.experiment_id).report()
        print(f"Report complete: {args.experiment_id}")
        return

    if args.command == "interpret":
        Experiment.from_output_dir(args.experiment_id).run_interpretation()
        return

    if args.command == "export-paper-bundle":
        Experiment.from_output_dir(args.experiment_id).export_manuscript_bundle()
        print(f"Bundle exported: {args.experiment_id}")
        return

    raise SystemExit(f"Unsupported command: {args.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clustro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("validate-config", "inspect-data", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("config")

    for name in ("resume", "status", "consensus", "interpret", "report", "export-paper-bundle"):
        command = subparsers.add_parser(name)
        command.add_argument("experiment_id")

    return parser
