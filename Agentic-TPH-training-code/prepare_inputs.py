#!/usr/bin/env python3
"""Convert source records into model inputs; use prepare to split the resulting data."""

import argparse
from pathlib import Path

from core_engine.trainer.workspaces import write_json


def build_parser() -> argparse.ArgumentParser:
    """Describe source directory layouts and required sequence reference files."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    hitph = commands.add_parser(
        "hitph", help="Combine positive records from the selected Hi-TpH data level"
    )
    hitph.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Hi-TpH source root containing benchmarks_dataset/level3 or level4",
    )
    hitph.add_argument(
        "--level",
        choices=("III", "IV"),
        required=True,
        help="III uses paired CDR3s; IV uses variable domains and retains original CDR3 identities",
    )
    hitph.add_argument(
        "--output", type=Path, required=True, help="New CSV file for positive records"
    )
    pan = commands.add_parser(
        "pan-source",
        help="Resolve positive TPH records and MHC reference chains across four domains",
    )
    pan.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="TPH_260630 root containing human/mouse/other Level III source files",
    )
    pan.add_argument(
        "--hla-fasta",
        type=Path,
        required=True,
        help="Human HLA protein reference FASTA",
    )
    pan.add_argument(
        "--mouse-fasta",
        type=Path,
        required=True,
        help="Mouse MHC protein reference FASTA from UniProt",
    )
    pan.add_argument(
        "--ipd-mhc-fasta",
        type=Path,
        required=True,
        help="IPD-MHC protein reference FASTA for other species",
    )
    pan.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for positive_records.csv, exclusions.csv and resolved sequence records",
    )
    return parser


def convert_inputs(args: argparse.Namespace) -> Path:
    """Read source files and write new inputs without modifying the source data."""
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == "hitph":
        from core_engine.trainer.hitph_input import load_hitph_identity

        frame = load_hitph_identity(args.root, args.level)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    else:
        from core_engine.trainer.pan_sources import (
            PanMHCSequenceResolver,
            prepare_pan_mhc_data,
        )

        resolver = PanMHCSequenceResolver(
            hla_fasta=args.hla_fasta,
            mouse_uniprot_fasta=args.mouse_fasta,
            ipd_mhc_fasta=args.ipd_mhc_fasta,
        )
        result = prepare_pan_mhc_data(data_root=args.data_root, resolver=resolver)
        args.output.mkdir(parents=True)
        result.positives.to_csv(args.output / "positive_records.csv", index=False)
        result.exclusions.to_csv(args.output / "exclusions.csv", index=False)
        result.sequence_registry.to_csv(
            args.output / "used_mhc_sequence_registry.csv", index=False
        )
        write_json(args.output / "source_report.json", result.report)
    return args.output


def main(argv: list[str] | None = None) -> None:
    """Convert inputs and print the output location."""
    print(convert_inputs(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
