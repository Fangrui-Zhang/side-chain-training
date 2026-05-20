"""Command line interface for DINA2."""

from __future__ import annotations

import argparse

from .embeddings import ESM2_LAYER, ESM2_MODEL, extract_esm2_embeddings
from .evaluation import evaluate_predictions
from .features import extract_features_from_manifest
from .labels import build_joined_dataset, import_cpmg_labels
from .splits import create_sequence_splits
from .training import train_model
from .ablations import run_ablations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dina2", description="DINA2 sidechain-aware dynamics pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-dyna1-labels", help="Import Dyna-1 CPMG labels with strict sequence mapping")
    p.add_argument("--label-zip", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-labels", required=True)
    p.add_argument("--out-qc", required=True)
    p.add_argument("--min-label-coverage", type=float, default=0.90)
    p.add_argument("--min-identity", type=float, default=0.90)
    p.add_argument("--include-unsuppressed", action="store_true")

    p = sub.add_parser("extract-sidechain-features", help="Extract residue sidechain features from ESMFold PDBs")
    p.add_argument("--manifest", required=True)
    p.add_argument("--pdb-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--out-qc", default=None)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("extract-esm2-embeddings", help="Pre-extract frozen ESM-2 embeddings")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default=ESM2_MODEL)
    p.add_argument("--layer", type=int, default=ESM2_LAYER)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("build-dataset", help="Join verified labels to feature rows")
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("make-splits", help="Create protein-level sequence-clustered splits")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cluster-tsv", default=None)
    p.add_argument("--eval-ids", default=None)
    p.add_argument("--min-seq-id", type=float, default=0.30)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--work-dir", default="data/mmseqs")

    p = sub.add_parser("train", help="Train a compact DINA2 baseline or fusion model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--embedding-index", default=None)
    p.add_argument("--model-type", choices=["embedding", "sidechain", "fusion"], default="fusion")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-missing", type=float, default=0.2)

    p = sub.add_parser("evaluate", help="Evaluate a prediction CSV")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", default=None)

    p = sub.add_parser("run-ablations", help="Run compact model/lambda ablations")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--embedding-index", default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "import-dyna1-labels":
        labels, qc = import_cpmg_labels(
            args.label_zip,
            args.manifest,
            args.out_labels,
            args.out_qc,
            min_label_coverage=args.min_label_coverage,
            min_identity=args.min_identity,
            include_unsuppressed=args.include_unsuppressed,
        )
        print(f"Wrote {len(labels)} label rows and {len(qc)} QC rows")
    elif args.command == "extract-sidechain-features":
        features, qc = extract_features_from_manifest(
            args.manifest,
            args.out,
            pdb_root=args.pdb_root,
            limit=args.limit,
            out_qc=args.out_qc,
        )
        print(f"Wrote {len(features)} feature rows and {len(qc)} QC rows")
    elif args.command == "extract-esm2-embeddings":
        index = extract_esm2_embeddings(args.manifest, args.out_dir, model_name=args.model_name, layer=args.layer, limit=args.limit)
        print(f"Wrote {len(index)} embedding files")
    elif args.command == "build-dataset":
        joined = build_joined_dataset(args.features, args.labels, args.out)
        print(f"Wrote {len(joined)} joined rows")
    elif args.command == "make-splits":
        split = create_sequence_splits(
            args.manifest,
            args.out,
            cluster_tsv=args.cluster_tsv,
            eval_ids_csv=args.eval_ids,
            min_seq_id=args.min_seq_id,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            work_dir=args.work_dir,
        )
        print(f"Wrote {len(split)} split rows")
    elif args.command == "train":
        result = train_model(
            args.dataset,
            args.embedding_index,
            args.split,
            args.out_dir,
            model_type=args.model_type,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_missing=args.lambda_missing,
        )
        print(f"Wrote model artifacts to {result['out_dir']}")
    elif args.command == "evaluate":
        metrics = evaluate_predictions(args.predictions, args.out)
        print(metrics.to_string(index=False))
    elif args.command == "run-ablations":
        index = run_ablations(
            args.dataset,
            args.split,
            args.out_dir,
            embedding_index_csv=args.embedding_index,
            epochs=args.epochs,
            seed=args.seed,
        )
        print(index.to_string(index=False))
    else:  # pragma: no cover
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
