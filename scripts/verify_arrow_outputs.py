#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

import pyarrow.parquet as pq
import polars as pl


def resolve_text_column(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from lib.config import get_config
        return get_config().text_column
    except Exception:
        return "text"


def count_nonempty_lines(path: Path, skip_header: bool = False) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        if skip_header:
            next(f, None)
        for line in f:
            if line.strip():
                count += 1
    return count


def resolve_defer_embedding_union() -> bool:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from lib.config import get_config
        return bool(get_config().defer_embedding_union)
    except Exception:
        return False


def count_csv_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        for _ in reader:
            count += 1
    return count


def load_test_texts(path: Path, text_column: str) -> tuple[int, set[str]] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or text_column not in reader.fieldnames:
            log(f"[ERROR] Missing text column '{text_column}' in {path}")
            return None
        texts = set()
        rows = 0
        for row in reader:
            rows += 1
            value = row.get(text_column)
            texts.add(str(value).strip())
        return rows, texts


def count_val_rows_with_overlap(path: Path, text_column: str, test_texts: set[str] | None) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or text_column not in reader.fieldnames:
            log(f"[ERROR] Missing text column '{text_column}' in {path}")
            raw_rows = count_csv_rows(path)
            return raw_rows, raw_rows, 0
        raw_rows = 0
        kept_rows = 0
        removed = 0
        for row in reader:
            raw_rows += 1
            value = row.get(text_column)
            text = str(value).strip() if value is not None else ""
            if test_texts is not None and text in test_texts:
                removed += 1
                continue
            kept_rows += 1
        return raw_rows, kept_rows, removed


def parquet_rows_cols(path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, pf.metadata.num_columns


def read_sparse_metadata(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    df = pl.read_parquet(str(path))
    if df.is_empty():
        return None
    if "shape_0" not in df.columns or "shape_1" not in df.columns:
        return None
    return int(df["shape_0"][0]), int(df["shape_1"][0])


def log(msg: str) -> None:
    print(msg)


def check_sparse(stage_dir: Path, name: str, expected_rows: int | None, expected_cols: int | None, errors: list[str]) -> tuple[int, int] | None:
    meta_path = stage_dir / f"{name}_metadata.parquet"
    meta = read_sparse_metadata(meta_path)
    if meta is None:
        errors.append(f"Missing or invalid metadata: {meta_path}")
        log(f"[ERROR] Missing/invalid metadata: {meta_path}")
        return None
    rows, cols = meta
    log(f"[OK] {meta_path.name}: rows={rows} cols={cols}")
    if expected_rows is not None and rows != expected_rows:
        errors.append(f"Row mismatch for {name}: rows={rows} expected={expected_rows}")
        log(f"[ERROR] Row mismatch for {name}: rows={rows} expected={expected_rows}")
    if expected_cols is not None and cols != expected_cols:
        errors.append(f"Col mismatch for {name}: cols={cols} expected={expected_cols}")
        log(f"[ERROR] Col mismatch for {name}: cols={cols} expected={expected_cols}")
    return rows, cols


def check_embeddings(stage_dir: Path, expected_rows: dict[str, int], errors: list[str]) -> None:
    for parquet_path in sorted(stage_dir.glob("embeddings_*.parquet")):
        name = parquet_path.stem
        dataset = None
        if name.startswith("embeddings_train"):
            dataset = "train"
        elif name.startswith("embeddings_val"):
            dataset = "val"
        elif name.startswith("embeddings_test"):
            dataset = "test"
        if dataset is None:
            log(f"[SKIP] {parquet_path.name}: unrecognized embedding prefix")
            continue
        rows, cols = parquet_rows_cols(parquet_path)
        expected = expected_rows.get(dataset)
        log(f"[OK] {parquet_path.name}: rows={rows} cols={cols} expected_rows={expected}")
        if expected is not None and rows != expected:
            errors.append(f"Embedding row mismatch for {parquet_path.name}: rows={rows} expected={expected}")
            log(f"[ERROR] Embedding row mismatch for {parquet_path.name}: rows={rows} expected={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Arrow outputs against expected row counts.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/val/test CSV files")
    parser.add_argument("--arrow-dir", default="outputs/arrow_data", help="Arrow outputs root directory")
    parser.add_argument("--jsonl", default=None, help="Optional original jsonl file to verify total count")
    parser.add_argument("--text-column", default=None, help="Text column name for overlap checks")
    parser.add_argument("--remove-val-test-duplicates", action="store_true", default=True, help="Apply val/test overlap removal (default)")
    parser.add_argument("--no-remove-val-test-duplicates", dest="remove_val_test_duplicates", action="store_false", help="Disable val/test overlap removal")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any mismatches are found")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    arrow_dir = Path(args.arrow_dir)

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    text_column = resolve_text_column(args.text_column)
    log(f"Text column for overlap checks: {text_column}")

    train_rows = count_csv_rows(train_csv) if train_csv.exists() else None
    test_rows = None
    test_texts = None
    if test_csv.exists():
        test_result = load_test_texts(test_csv, text_column)
        if test_result is not None:
            test_rows, test_texts = test_result
        else:
            test_rows = count_csv_rows(test_csv)
    val_rows_raw = None
    val_rows_effective = None
    val_removed = 0
    if val_csv.exists():
        if args.remove_val_test_duplicates and test_texts is not None:
            val_rows_raw, val_rows_effective, val_removed = count_val_rows_with_overlap(val_csv, text_column, test_texts)
        else:
            val_rows_raw = count_csv_rows(val_csv)
            val_rows_effective = val_rows_raw
    val_rows = val_rows_effective

    log(f"Train rows: {train_rows}")
    log(f"Val rows (raw): {val_rows_raw}")
    if args.remove_val_test_duplicates:
        log(f"Val rows (after val/test overlap removal): {val_rows_effective} (removed {val_removed})")
    else:
        log("Val/test overlap removal disabled")
    log(f"Test rows: {test_rows}")
    if train_rows is not None and val_rows_raw is not None and test_rows is not None:
        log(f"Total rows (csv raw): {train_rows + val_rows_raw + test_rows}")
    if train_rows is not None and val_rows is not None and test_rows is not None:
        log(f"Total rows (csv effective): {train_rows + val_rows + test_rows}")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if jsonl_path.exists():
            jsonl_rows = count_nonempty_lines(jsonl_path, skip_header=False)
            log(f"Original jsonl rows: {jsonl_rows}")
            if train_rows is not None and val_rows is not None and test_rows is not None:
                total_effective = train_rows + val_rows + test_rows
                total_raw = train_rows + (val_rows_raw or 0) + test_rows
                if total_effective != jsonl_rows and total_raw != jsonl_rows:
                    log(f"[ERROR] jsonl total mismatch: csv_effective={total_effective} csv_raw={total_raw} jsonl_rows={jsonl_rows}")
                elif total_raw == jsonl_rows and total_effective != jsonl_rows:
                    log("[OK] jsonl matches raw total; effective total lower due to overlap removal")
                else:
                    log("[OK] jsonl total matches csv total")
        else:
            log(f"[ERROR] jsonl not found: {jsonl_path}")

    errors: list[str] = []
    stage_3 = arrow_dir / "stage_3"
    stage_4 = arrow_dir / "stage_4"
    stage_5 = arrow_dir / "stage_5"

    if stage_3.exists():
        log(f"Checking stage 3 sparse outputs in {stage_3}...")
        train_meta = check_sparse(stage_3, "train_nlp_checkpoint", train_rows, None, errors)
        val_meta = check_sparse(stage_3, "val_nlp_checkpoint", val_rows, train_meta[1] if train_meta else None, errors)
        test_meta = check_sparse(stage_3, "test_nlp_checkpoint", test_rows, train_meta[1] if train_meta else None, errors)
        if resolve_defer_embedding_union():
            log("Skipping stage 3 feature matrix checks (defer_embedding_union enabled)")
        else:
            train_feat = check_sparse(stage_3, "train_features", train_rows, None, errors)
            check_sparse(stage_3, "val_features", val_rows, train_feat[1] if train_feat else None, errors)
            check_sparse(stage_3, "test_features", test_rows, train_feat[1] if train_feat else None, errors)
        log("Checking stage 3 embeddings in stage_3...")
        check_embeddings(stage_3, {"train": train_rows, "val": val_rows, "test": test_rows}, errors)
    else:
        log(f"[SKIP] stage_3 not found at {stage_3}")

    if stage_4.exists():
        log(f"Checking stage 4 sparse outputs in {stage_4}...")
        train_proc = check_sparse(stage_4, "train_preprocessed", train_rows, None, errors)
        check_sparse(stage_4, "val_preprocessed", val_rows, train_proc[1] if train_proc else None, errors)
        check_sparse(stage_4, "test_preprocessed", test_rows, train_proc[1] if train_proc else None, errors)
    else:
        log(f"[SKIP] stage_4 not found at {stage_4}")

    if stage_5.exists():
        log(f"Checking stage 5 sparse outputs in {stage_5}...")
        train_rfe = check_sparse(stage_5, "train_features_selected", train_rows, None, errors)
        check_sparse(stage_5, "val_features_selected", val_rows, train_rfe[1] if train_rfe else None, errors)
        check_sparse(stage_5, "test_features_selected", test_rows, train_rfe[1] if train_rfe else None, errors)
    else:
        log(f"[SKIP] stage_5 not found at {stage_5}")

    if errors and args.strict:
        log("Errors detected. Exiting with non-zero status.")
        return 1
    if errors:
        log("Errors detected. See above.")
        return 0
    log("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
