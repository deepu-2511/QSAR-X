#!/usr/bin/env python3
"""
Brucellosis Natural-Compound Network Pharmacology Workflow
Based on Task 25 methodology:
1) disease targets, 2) natural compounds, 3) compound targets,
4) overlapping targets, 5) PPI network, 6) optional AutoDock Vina scoring.

Important:
- This script does NOT invent disease targets or compound-target associations.
- Prepare the CSV input files from reliable databases/literature and record source URLs.
- STRING PPI retrieval needs internet access.
- Docking requires prepared receptor/ligand PDBQT files and AutoDock Vina installed.
"""

from pathlib import Path
import argparse
import sys

try:
    import pandas as pd
except ImportError:
    sys.exit("Missing dependency: pandas. Install with: python -m pip install pandas")

OUT = Path("results")
OUT.mkdir(exist_ok=True)

def read_csv(path, required):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required input file not found: {p.resolve()}")
    df = pd.read_csv(p)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing columns: {', '.join(missing)}")
    return df

def normalize_targets(df, col):
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    out = out[out[col].ne("") & out[col].ne("nan")]
    return out.drop_duplicates(subset=[col])

def overlap_analysis(disease_file, compound_targets_file):
    disease = normalize_targets(read_csv(disease_file, ["target"]), "target")
    ct = read_csv(compound_targets_file, ["compound", "target"])
    ct["compound"] = ct["compound"].astype(str).str.strip()
    ct["target"] = ct["target"].astype(str).str.strip()
    ct = ct[(ct["compound"] != "") & (ct["target"] != "")]
    ct = ct.drop_duplicates(subset=["compound", "target"])

    disease_set = set(disease["target"].str.upper())
    ct["target_upper"] = ct["target"].str.upper()
    common = ct[ct["target_upper"].isin(disease_set)].drop(columns=["target_upper"])
    common.to_csv(OUT / "common_targets_by_compound.csv", index=False)

    common_targets = sorted(set(common["target"].astype(str)))
    pd.DataFrame({"target": common_targets}).to_csv(OUT / "common_targets.csv", index=False)

    summary = (common.groupby("compound")["target"]
               .agg(lambda s: "; ".join(sorted(set(map(str, s)))))
               .reset_index(name="common_targets"))
    summary["number_of_common_targets"] = summary["common_targets"].apply(
        lambda s: 0 if not s else len(s.split("; ")))
    summary.to_csv(OUT / "compound_overlap_summary.csv", index=False)

    print(f"Disease targets: {len(disease)}")
    print(f"Compound-target associations: {len(ct)}")
    print(f"Common target genes/proteins: {len(common_targets)}")
    print(f"Results written to: {OUT.resolve()}")
    return common_targets

def string_ppi(common_targets, species_id, score, output_prefix="ppi"):
    try:
        import requests
    except ImportError:
        print("Skipping STRING PPI: install requests with `python -m pip install requests`.")
        return

    if not common_targets:
        print("No common targets found; skipping PPI.")
        return

    # STRING accepts protein names or gene symbols. Verify organism/species ID.
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0d".join(common_targets),
        "species": int(species_id),
        "required_score": int(score),
        "network_type": "functional"
    }
    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            print("STRING returned no network edges. Check target identifiers/species ID.")
            return
        (OUT / f"{output_prefix}_string_edges.tsv").write_text(text + "\n", encoding="utf-8")
        print(f"STRING network edges saved: {OUT / f'{output_prefix}_string_edges.tsv'}")
    except Exception as exc:
        print(f"STRING request failed: {exc}")
        print("Check internet access, gene symbols, and species ID.")

def analyze_ppi(edge_file):
    try:
        import networkx as nx
    except ImportError:
        print("Skipping PPI centrality: install networkx with `python -m pip install networkx`.")
        return
    p = Path(edge_file)
    if not p.exists():
        print(f"Skipping centrality: {p} not found.")
        return
    try:
        df = pd.read_csv(p, sep="\t")
        # STRING network TSV commonly uses preferredName_A and preferredName_B.
        if not {"preferredName_A", "preferredName_B"}.issubset(df.columns):
            print("Could not find STRING edge columns preferredName_A and preferredName_B.")
            print("Inspect the TSV and adapt column names if needed.")
            return
        g = nx.from_pandas_edgelist(df, "preferredName_A", "preferredName_B",
                                    edge_attr="score")
        degree = nx.degree_centrality(g)
        betweenness = nx.betweenness_centrality(g)
        rows = [{
            "protein": node,
            "degree": g.degree(node),
            "degree_centrality": degree[node],
            "betweenness_centrality": betweenness[node]
        } for node in g.nodes()]
        result = pd.DataFrame(rows).sort_values(
            ["degree_centrality", "betweenness_centrality"], ascending=False)
        result.to_csv(OUT / "ppi_hub_candidates.csv", index=False)
        print(f"PPI hub candidates saved: {OUT / 'ppi_hub_candidates.csv'}")
    except Exception as exc:
        print(f"Could not analyze PPI file: {exc}")

def docking_scores(docking_csv):
    """Summarize pre-computed Vina scores from a CSV; it does not prepare structures."""
    df = read_csv(docking_csv, ["compound", "target", "vina_score_kcal_mol"])
    df["vina_score_kcal_mol"] = pd.to_numeric(df["vina_score_kcal_mol"], errors="coerce")
    df = df.dropna(subset=["vina_score_kcal_mol"]).sort_values("vina_score_kcal_mol")
    df.to_csv(OUT / "docking_ranked.csv", index=False)
    print("Docking scores ranked (more negative scores indicate more favorable predicted binding,")
    print("but scores alone do not prove inhibition or biological activity).")
    print(f"Saved: {OUT / 'docking_ranked.csv'}")

def make_templates():
    pd.DataFrame(columns=["target", "organism", "database", "source_url", "evidence"]).to_csv(
        "disease_targets_template.csv", index=False)
    pd.DataFrame(columns=["compound", "target", "target_organism", "database",
                          "source_url", "evidence"]).to_csv(
        "compound_targets_template.csv", index=False)
    pd.DataFrame(columns=["compound", "plant_or_natural_source", "smiles",
                          "pubchem_cid", "source_url", "literature_evidence"]).to_csv(
        "natural_compounds_template.csv", index=False)
    pd.DataFrame(columns=["compound", "target", "vina_score_kcal_mol",
                          "receptor_file", "ligand_file", "notes"]).to_csv(
        "docking_scores_template.csv", index=False)
    print("Created CSV templates in the current folder.")

def main():
    parser = argparse.ArgumentParser(
        description="Network pharmacology workflow for brucellosis natural compounds.")
    parser.add_argument("--templates", action="store_true",
                        help="Create blank CSV input templates and exit.")
    parser.add_argument("--disease", default="disease_targets.csv",
                        help="CSV containing a 'target' column.")
    parser.add_argument("--compound-targets", default="compound_targets.csv",
                        help="CSV containing 'compound' and 'target' columns.")
    parser.add_argument("--string", action="store_true",
                        help="Query STRING for PPI edges after finding overlap.")
    parser.add_argument("--species-id", type=int, default=9606,
                        help="NCBI taxonomy ID (9606 human; 234 Brucella abortus).")
    parser.add_argument("--string-score", type=int, default=700,
                        help="Minimum STRING score 0-1000; default 700 (high confidence).")
    parser.add_argument("--analyze-ppi", default=None,
                        help="Analyze an existing STRING edge TSV file.")
    parser.add_argument("--docking-csv", default=None,
                        help="Optional CSV of already-computed AutoDock Vina scores.")
    args = parser.parse_args()

    if args.templates:
        make_templates()
        return

    common = overlap_analysis(args.disease, args.compound_targets)
    if args.string:
        string_ppi(common, args.species_id, args.string_score)
        analyze_ppi(OUT / "ppi_string_edges.tsv")
    if args.analyze_ppi:
        analyze_ppi(args.analyze_ppi)
    if args.docking_csv:
        docking_scores(args.docking_csv)

    print("\nNext steps:")
    print("1. Review common targets and confirm species/target identity.")
    print("2. Visualize and interpret the PPI network; do not treat centrality as proof of causality.")
    print("3. Select biologically justified targets and obtain experimentally supported protein structures.")
    print("4. Prepare ligand/receptor PDBQT files, define the docking box, then run AutoDock Vina.")
    print("5. Inspect poses/interactions and validate findings with literature/controls.")
    print("6. Agonist/antagonist claims require functional evidence; docking alone cannot establish them.")

if __name__ == "__main__":
    main()
