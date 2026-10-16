from chembl_webresource_client.new_client import new_client
import pandas as pd
import sys
import time


def find_targets(target_name):
    target = new_client.target
    results = target.search(target_name)

    rows = []

    for item in results:
        rows.append({
            "target_chembl_id": item.get("target_chembl_id"),
            "pref_name": item.get("pref_name"),
            "target_type": item.get("target_type"),
            "organism": item.get("organism"),
        })

    return pd.DataFrame(rows)


def download_activity(target_id, max_records=5000):

    print()
    print("Requesting activity data...")
    print(f"Maximum records for this test: {max_records}")
    print()

    activity = new_client.activity

    records = activity.filter(
        target_chembl_id=target_id,
        standard_type="IC50",
        standard_relation="=",
        standard_units="nM",
        standard_value__isnull=False,
        molecule_chembl_id__isnull=False,
    ).only([
        "activity_id",
        "molecule_chembl_id",
        "target_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_chembl_id",
        "assay_type",
    ])

    rows = []

    try:

        for count, record in enumerate(records, start=1):

            rows.append(dict(record))

            if count % 100 == 0:
                print(f"Downloaded activity records: {count}")

            if count >= max_records:
                break

    except KeyboardInterrupt:

        print()
        print("Download interrupted.")
        print(f"Records collected before interruption: {len(rows)}")

    except Exception as e:

        print()
        print("ChEMBL request error:")
        print(type(e).__name__)
        print(str(e))

    return pd.DataFrame(rows)


def download_smiles(df):

    molecule = new_client.molecule

    ids = (
        df["molecule_chembl_id"]
        .dropna()
        .unique()
        .tolist()
    )

    structures = {}

    batch_size = 25

    total = len(ids)

    print()
    print(f"Unique compounds requiring SMILES: {total}")
    print()

    for i in range(0, total, batch_size):

        batch = ids[i:i + batch_size]

        print(
            f"Downloading structures "
            f"{i + 1}-{min(i + batch_size, total)} "
            f"of {total}..."
        )

        try:

            results = molecule.filter(
                molecule_chembl_id__in=batch
            ).only([
                "molecule_chembl_id",
                "molecule_structures"
            ])

            for item in results:

                mol_id = item.get(
                    "molecule_chembl_id"
                )

                mol_structures = item.get(
                    "molecule_structures"
                )

                smiles = None

                if mol_structures:
                    smiles = mol_structures.get(
                        "canonical_smiles"
                    )

                structures[mol_id] = smiles

        except Exception as e:

            print(
                f"Warning: batch failed: {e}"
            )

        time.sleep(0.2)

    df["smiles"] = df[
        "molecule_chembl_id"
    ].map(structures)

    return df


def main():

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "python 01_download_chembl.py EGFR"
        )

        return

    target_name = " ".join(
        sys.argv[1:]
    )

    print()
    print("=" * 60)
    print("QSAR DATA ACQUISITION - ChEMBL")
    print("=" * 60)
    print()

    print(
        f"Searching ChEMBL targets for: "
        f"{target_name}"
    )

    print()

    targets = find_targets(
        target_name
    )

    if targets.empty:

        print("No targets found.")
        return

    print(
        targets.to_string(
            index=False
        )
    )

    print()
    print("=" * 60)
    print("Select the target")
    print("=" * 60)
    print()

    target_id = input(
        "Enter Target ChEMBL ID: "
    ).strip()

    print()
    print(
        f"Selected target: {target_id}"
    )

    df = download_activity(
        target_id,
        max_records=5000
    )

    if df.empty:

        print()
        print(
            "No activity records were downloaded."
        )

        return

    print()
    print(
        f"Activity records collected: "
        f"{len(df)}"
    )

    print()
    print(
        "Downloading compound structures..."
    )

    df = download_smiles(df)

    before = len(df)

    df = df.dropna(
        subset=["smiles"]
    )

    print()
    print(
        f"Records before SMILES filtering: "
        f"{before}"
    )

    print(
        f"Records with SMILES: "
        f"{len(df)}"
    )

    output = (
        f"{target_id}_IC50_raw.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print()
    print(f"Saved: {output}")
    print(
        f"Final records: {len(df)}"
    )
    print()


if __name__ == "__main__":
    main()