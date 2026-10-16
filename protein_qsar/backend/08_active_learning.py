
"""
QSAR-X Professional — Active Learning Candidate Selector
Uses the existing predictor.py without replacing the production model.
"""

import argparse
import heapq
from itertools import count
from pathlib import Path

import pandas as pd

from predictor import predict


OUTPUT_DIR = Path(__file__).resolve().parent
SUPPORTED_SMILES_COLUMNS = {
    "smiles",
    "canonical_smiles",
    "structure",
    "molecule_smiles",
}


def find_smiles_column(columns):
    for column in columns:
        if str(column).strip().lower() in SUPPORTED_SMILES_COLUMNS:
            return column
    raise ValueError(
        "CSV must contain a smiles, canonical_smiles, "
        "structure, or molecule_smiles column."
    )


def run_active_learning(input_file, top_n=10000, chunk_size=2000):
    input_path = Path(input_file).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if top_n < 1 or chunk_size < 1:
        raise ValueError("top_n and chunk_size must be positive.")

    predictions_file = OUTPUT_DIR / "active_learning_predictions.csv"
    queue_file = OUTPUT_DIR / "active_learning_queue.csv"

    # Do not silently append results from an older run.
    predictions_file.unlink(missing_ok=True)
    queue_file.unlink(missing_ok=True)

    # Min-heap retains only the top-ranked candidates in memory.
    best = []
    sequence = count()

    total = successful = failed = 0

    for chunk in pd.read_csv(
        input_path,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    ):
        smiles_col = find_smiles_column(chunk.columns)
        records = []

        for raw_smiles in chunk[smiles_col]:
            total += 1
            smiles = str(raw_smiles).strip()

            if not smiles:
                failed += 1
                records.append({
                    "input_smiles": smiles,
                    "status": "FAILED",
                    "error": "Empty SMILES",
                })
                continue

            try:
                result = predict(smiles)

                pred = result["prediction"]
                unc = result["uncertainty"]
                ad = result["applicability_domain"]

                uncertainty = float(unc["prediction_sd"])
                similarity = float(ad["max_tanimoto_similarity"])

                if not (
                    pd.notna(uncertainty)
                    and pd.notna(similarity)
                ):
                    raise ValueError("Non-finite uncertainty or similarity")

                import math
                if not (
                    math.isfinite(uncertainty)
                    and math.isfinite(similarity)
                ):
                    raise ValueError("Non-finite uncertainty or similarity")

                # Heuristic prioritization score, NOT a probability.
                # High model disagreement and low reference similarity
                # increase priority for review.
                score = uncertainty + 0.5 * (1.0 - similarity)

                record = {
                    "input_smiles": smiles,
                    "canonical_smiles":
                        result["compound"]["canonical_smiles"],
                    "predicted_pIC50": pred["pIC50"],
                    "predicted_IC50_nM": pred["IC50_nM"],
                    "uncertainty": uncertainty,
                    "max_tanimoto_similarity": similarity,
                    "applicability_domain": ad["status"],
                    "confidence": result["confidence"],
                    "selection_score": score,
                    "status": "SUCCESS",
                    "error": "",
                }

                successful += 1
                item_id = next(sequence)
                item = (score, item_id, record)

                if len(best) < top_n:
                    heapq.heappush(best, item)
                elif score > best[0][0]:
                    heapq.heapreplace(best, item)

                records.append(record)

            except Exception as exc:
                failed += 1
                records.append({
                    "input_smiles": smiles,
                    "status": "FAILED",
                    "error": str(exc)[:300],
                })

        pd.DataFrame(records).to_csv(
            predictions_file,
            mode="a",
            index=False,
            header=not predictions_file.exists(),
        )

        print(
            f"Processed {total:,} | "
            f"Successful {successful:,} | Failed {failed:,}"
        )

    ranked = sorted(best, key=lambda item: item[0], reverse=True)
    queue_records = [item[2] for item in ranked]

    pd.DataFrame(queue_records).to_csv(queue_file, index=False)

    print("\nActive Learning selection finished.")
    print(f"Total rows: {total:,}")
    print(f"Successful predictions: {successful:,}")
    print(f"Failed predictions: {failed:,}")
    print(f"All predictions: {predictions_file}")
    print(f"Ranked candidate queue: {queue_file}")
    print(
        "\nNote: The queue prioritizes uncertainty and novelty. "
        "It does not train the model or prove experimental activity."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input compounds CSV")
    parser.add_argument("--top-n", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    args = parser.parse_args()

    run_active_learning(
        args.input,
        top_n=args.top_n,
        chunk_size=args.chunk_size,
    )
