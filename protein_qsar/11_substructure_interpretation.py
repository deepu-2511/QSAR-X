import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator


# ============================================================
# STEP 11
# MOLECULAR SUBSTRUCTURE INTERPRETATION
# + SHAP FINGERPRINT BIT MAPPING
# ============================================================

print("=" * 80)
print("STEP 11 — MOLECULAR SUBSTRUCTURE INTERPRETATION")
print("=" * 80)


# ============================================================
# CONFIGURATION
# ============================================================

SHAP_FILE = os.path.join(
    "step10_results",
    "step10_shap_global_importance.csv"
)

TEST_FILE = "CHEMBL203_test.csv"

OUTPUT_DIR = "step11_results"

FP_RADIUS = 2
FP_BITS = 2048
TOP_BITS = 20

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "substructure_images"
)

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(SHAP_FILE):
    raise FileNotFoundError(
        f"SHAP file not found:\n{SHAP_FILE}"
    )

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_FILE}"
    )

print(f"OK: {SHAP_FILE}")
print(f"OK: {TEST_FILE}")


# ============================================================
# LOAD STEP 10 SHAP RESULTS
# ============================================================

print("\nLoading Step 10 SHAP results...")

shap_df = pd.read_csv(
    SHAP_FILE
)

print(
    f"Total fingerprint features: "
    f"{len(shap_df)}"
)


# ============================================================
# CHECK SHAP COLUMNS
# ============================================================

required_shap_columns = [
    "feature",
    "mean_abs_shap"
]

for column in required_shap_columns:

    if column not in shap_df.columns:

        raise ValueError(
            f"Missing column in SHAP file: "
            f"{column}"
        )


# ============================================================
# SELECT TOP SHAP BITS
# ============================================================

top_shap = (
    shap_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .head(TOP_BITS)
    .copy()
)


# Convert feature name to numeric bit ID

top_shap["bit_id"] = (
    top_shap["feature"]
    .str.replace(
        "ECFP4_bit_",
        "",
        regex=False
    )
    .astype(int)
)


# Save top SHAP bits

top_shap_file = os.path.join(
    OUTPUT_DIR,
    "step11_top_shap_bits.csv"
)

top_shap.to_csv(
    top_shap_file,
    index=False
)


print("\nTop SHAP bits:")

print(
    top_shap[
        [
            "bit_id",
            "feature",
            "mean_abs_shap"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"Test compounds: "
    f"{len(test_df)}"
)


# ============================================================
# CHECK TEST DATA
# ============================================================

if "canonical_smiles" not in test_df.columns:

    raise ValueError(
        "Column 'canonical_smiles' "
        "was not found in CHEMBL203_test.csv"
    )


# ============================================================
# CREATE MORGAN GENERATOR
# ============================================================

print(
    "\nCreating RDKit MorganGenerator..."
)

morgan_generator = (
    rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_BITS
    )
)


# ============================================================
# FUNCTION 1
# GET MORGAN BIT INFORMATION
# ============================================================

def get_bit_info(mol):

    """
    Generate Morgan fingerprint and retrieve
    atom/radius information for active bits.

    Compatible with the current RDKit
    MorganGenerator AdditionalOutput API.
    """

    additional_output = (
        rdFingerprintGenerator.AdditionalOutput()
    )

    additional_output.CollectBitInfoMap()

    morgan_generator.GetFingerprint(
        mol,
        additionalOutput=additional_output
    )

    bit_info = (
        additional_output.GetBitInfoMap()
    )

    return bit_info


# ============================================================
# FUNCTION 2
# EXTRACT MOLECULAR ENVIRONMENT
# ============================================================

def extract_environment(
    mol,
    atom_id,
    radius
):

    """
    Extract the molecular environment represented
    by a Morgan fingerprint bit.
    """

    try:

        bond_ids = (
            Chem.FindAtomEnvironmentOfRadiusN(
                mol,
                radius,
                atom_id
            )
        )

        atom_indices = {
            atom_id
        }

        for bond_id in bond_ids:

            bond = mol.GetBondWithIdx(
                bond_id
            )

            atom_indices.add(
                bond.GetBeginAtomIdx()
            )

            atom_indices.add(
                bond.GetEndAtomIdx()
            )

        atom_indices = sorted(
            atom_indices
        )

        bond_ids = list(
            bond_ids
        )

        substructure_smiles = (
            Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=atom_indices,
                canonical=True
            )
        )

        return (
            substructure_smiles,
            atom_indices,
            bond_ids
        )

    except Exception:

        atom = mol.GetAtomWithIdx(
            atom_id
        )

        return (
            atom.GetSymbol(),
            [atom_id],
            []
        )


# ============================================================
# MAP TOP SHAP BITS
# ============================================================

print(
    "\nMapping SHAP bits to molecular environments..."
)

mapping_records = []

valid_molecules = 0
invalid_molecules = 0


for compound_index, row in test_df.iterrows():

    smiles = row[
        "canonical_smiles"
    ]

    mol = Chem.MolFromSmiles(
        smiles
    )

    # --------------------------------------------------------
    # Invalid molecule
    # --------------------------------------------------------

    if mol is None:

        invalid_molecules += 1

        continue


    valid_molecules += 1


    # --------------------------------------------------------
    # Get Morgan bit information
    # --------------------------------------------------------

    bit_info = get_bit_info(
        mol
    )


    # --------------------------------------------------------
    # Examine top SHAP bits
    # --------------------------------------------------------

    for bit_id in top_shap[
        "bit_id"
    ]:

        bit_id = int(
            bit_id
        )


        # Bit absent in this molecule

        if bit_id not in bit_info:

            continue


        # One bit may have multiple environments

        environments = bit_info[
            bit_id
        ]


        for atom_id, radius in environments:

            atom = mol.GetAtomWithIdx(
                atom_id
            )

            central_atom = (
                atom.GetSymbol()
            )


            (
                substructure_smiles,
                atom_indices,
                bond_indices
            ) = extract_environment(
                mol,
                atom_id,
                radius
            )


            mapping_records.append(
                {
                    "test_index":
                        compound_index,

                    "molecule_chembl_id":
                        row.get(
                            "molecule_chembl_id",
                            ""
                        ),

                    "smiles":
                        smiles,

                    "bit_id":
                        bit_id,

                    "shap_feature":
                        f"ECFP4_bit_{bit_id}",

                    "atom_id":
                        atom_id,

                    "radius":
                        radius,

                    "central_atom":
                        central_atom,

                    "substructure_smiles":
                        substructure_smiles,

                    "atom_indices":
                        ",".join(
                            str(x)
                            for x in atom_indices
                        ),

                    "bond_indices":
                        ",".join(
                            str(x)
                            for x in bond_indices
                        )
                }
            )


# ============================================================
# SAVE BIT-SUBSTRUCTURE MAPPING
# ============================================================

mapping_df = pd.DataFrame(
    mapping_records
)


mapping_file = os.path.join(
    OUTPUT_DIR,
    "step11_bit_substructure_mapping.csv"
)


mapping_df.to_csv(
    mapping_file,
    index=False
)


print(
    f"\nValid molecules: "
    f"{valid_molecules}"
)

print(
    f"Invalid molecules: "
    f"{invalid_molecules}"
)

print(
    f"Mapped environments: "
    f"{len(mapping_df)}"
)

print(
    f"Saved: "
    f"{mapping_file}"
)


# ============================================================
# SELECT REPRESENTATIVE MOLECULES
# ============================================================

print(
    "\nSelecting representative molecules..."
)

representative_records = []


for bit_id in top_shap[
    "bit_id"
]:

    bit_id = int(
        bit_id
    )


    bit_rows = mapping_df[
        mapping_df[
            "bit_id"
        ] == bit_id
    ]


    if len(bit_rows) == 0:

        continue


    # Select up to three unique molecules

    selected = (
        bit_rows
        .drop_duplicates(
            subset=[
                "smiles"
            ]
        )
        .head(3)
    )


    for _, row in selected.iterrows():

        representative_records.append(
            {
                "bit_id":
                    bit_id,

                "feature":
                    row[
                        "shap_feature"
                    ],

                "test_index":
                    row[
                        "test_index"
                    ],

                "molecule_chembl_id":
                    row[
                        "molecule_chembl_id"
                    ],

                "smiles":
                    row[
                        "smiles"
                    ],

                "atom_id":
                    row[
                        "atom_id"
                    ],

                "radius":
                    row[
                        "radius"
                    ],

                "central_atom":
                    row[
                        "central_atom"
                    ],

                "substructure_smiles":
                    row[
                        "substructure_smiles"
                    ]
            }
        )


representative_df = pd.DataFrame(
    representative_records
)


representative_file = os.path.join(
    OUTPUT_DIR,
    "step11_representative_molecules.csv"
)


representative_df.to_csv(
    representative_file,
    index=False
)


print(
    f"Representative molecules: "
    f"{len(representative_df)}"
)

print(
    f"Saved: "
    f"{representative_file}"
)


# ============================================================
# GENERATE HIGHLIGHTED MOLECULE IMAGES
# ============================================================

print(
    "\nGenerating representative molecular images..."
)

image_records = []


for bit_id in top_shap[
    "bit_id"
]:

    bit_id = int(
        bit_id
    )


    bit_rows = representative_df[
        representative_df[
            "bit_id"
        ] == bit_id
    ]


    if len(bit_rows) == 0:

        continue


    for rank, (_, row) in enumerate(
        bit_rows.iterrows(),
        start=1
    ):

        mol = Chem.MolFromSmiles(
            row[
                "smiles"
            ]
        )


        if mol is None:

            continue


        atom_id = int(
            row[
                "atom_id"
            ]
        )

        radius = int(
            row[
                "radius"
            ]
        )


        # ----------------------------------------------------
        # Determine highlighted atoms and bonds
        # ----------------------------------------------------

        try:

            bond_ids = (
                Chem.FindAtomEnvironmentOfRadiusN(
                    mol,
                    radius,
                    atom_id
                )
            )


            highlight_bonds = list(
                bond_ids
            )


            highlight_atoms = {
                atom_id
            }


            for bond_id in bond_ids:

                bond = mol.GetBondWithIdx(
                    bond_id
                )


                highlight_atoms.add(
                    bond.GetBeginAtomIdx()
                )


                highlight_atoms.add(
                    bond.GetEndAtomIdx()
                )


            highlight_atoms = list(
                highlight_atoms
            )


        except Exception:

            highlight_atoms = [
                atom_id
            ]

            highlight_bonds = []


        # ----------------------------------------------------
        # Create image filename
        # ----------------------------------------------------

        image_name = (
            f"bit_{bit_id}_"
            f"example_{rank}.png"
        )


        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )


        # ----------------------------------------------------
        # Draw molecule
        # ----------------------------------------------------

        try:

            image = Draw.MolToImage(
                mol,
                size=(600, 450),
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds
            )


            image.save(
                image_path
            )


            image_records.append(
                {
                    "bit_id":
                        bit_id,

                    "rank":
                        rank,

                    "image":
                        image_path,

                    "smiles":
                        row[
                            "smiles"
                        ],

                    "substructure_smiles":
                        row[
                            "substructure_smiles"
                        ]
                }
            )


        except Exception as error:

            print(
                f"Image generation error "
                f"for bit {bit_id}: "
                f"{error}"
            )


image_df = pd.DataFrame(
    image_records
)


image_file = os.path.join(
    OUTPUT_DIR,
    "step11_generated_images.csv"
)


image_df.to_csv(
    image_file,
    index=False
)


print(
    f"Generated images: "
    f"{len(image_df)}"
)


# ============================================================
# BIT FREQUENCY ANALYSIS
# ============================================================

print(
    "\nCalculating bit occurrence frequency..."
)

frequency_records = []


for bit_id in top_shap[
    "bit_id"
]:

    bit_id = int(
        bit_id
    )


    rows = mapping_df[
        mapping_df[
            "bit_id"
        ] == bit_id
    ]


    shap_value = top_shap.loc[
        top_shap[
            "bit_id"
        ] == bit_id,
        "mean_abs_shap"
    ].iloc[0]


    unique_compounds = (
        rows[
            "test_index"
        ]
        .nunique()
    )


    unique_substructures = (
        rows[
            "substructure_smiles"
        ]
        .nunique()
    )


    frequency_records.append(
        {
            "bit_id":
                bit_id,

            "feature":
                f"ECFP4_bit_{bit_id}",

            "mean_abs_shap":
                shap_value,

            "environment_count":
                len(rows),

            "unique_compounds":
                unique_compounds,

            "unique_substructures":
                unique_substructures
        }
    )


frequency_df = pd.DataFrame(
    frequency_records
)


frequency_df = (
    frequency_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)


frequency_file = os.path.join(
    OUTPUT_DIR,
    "step11_bit_frequency.csv"
)


frequency_df.to_csv(
    frequency_file,
    index=False
)


# ============================================================
# SUBSTRUCTURE SUMMARY
# ============================================================

print(
    "\nCreating substructure summary..."
)

substructure_summary_records = []


for bit_id in top_shap[
    "bit_id"
]:

    bit_id = int(
        bit_id
    )


    rows = mapping_df[
        mapping_df[
            "bit_id"
        ] == bit_id
    ]


    shap_value = top_shap.loc[
        top_shap[
            "bit_id"
        ] == bit_id,
        "mean_abs_shap"
    ].iloc[0]


    if len(rows) > 0:

        substructure_counts = (
            rows[
                "substructure_smiles"
            ]
            .value_counts()
        )


        most_common_substructure = (
            substructure_counts
            .index[0]
        )


        most_common_count = (
            substructure_counts
            .iloc[0]
        )


    else:

        most_common_substructure = ""

        most_common_count = 0


    substructure_summary_records.append(
        {
            "bit_id":
                bit_id,

            "feature":
                f"ECFP4_bit_{bit_id}",

            "mean_abs_shap":
                shap_value,

            "mapped_environment_count":
                len(rows),

            "unique_compounds":
                rows[
                    "test_index"
                ].nunique(),

            "unique_substructures":
                rows[
                    "substructure_smiles"
                ].nunique(),

            "most_common_substructure":
                most_common_substructure,

            "most_common_count":
                most_common_count
        }
    )


substructure_summary_df = pd.DataFrame(
    substructure_summary_records
)


substructure_summary_file = os.path.join(
    OUTPUT_DIR,
    "step11_substructure_summary.csv"
)


substructure_summary_df.to_csv(
    substructure_summary_file,
    index=False
)


# ============================================================
# GENERAL SUMMARY
# ============================================================

summary_records = [

    {
        "parameter":
            "Fingerprint",
        "value":
            "Morgan / ECFP4"
    },

    {
        "parameter":
            "Fingerprint radius",
        "value":
            FP_RADIUS
    },

    {
        "parameter":
            "Fingerprint bits",
        "value":
            FP_BITS
    },

    {
        "parameter":
            "Test compounds",
        "value":
            len(test_df)
    },

    {
        "parameter":
            "Valid compounds",
        "value":
            valid_molecules
    },

    {
        "parameter":
            "Invalid compounds",
        "value":
            invalid_molecules
    },

    {
        "parameter":
            "Top SHAP bits analyzed",
        "value":
            TOP_BITS
    },

    {
        "parameter":
            "Mapped environments",
        "value":
            len(mapping_df)
    },

    {
        "parameter":
            "Representative molecules",
        "value":
            len(representative_df)
    },

    {
        "parameter":
            "Generated images",
        "value":
            len(image_df)
    }
]


summary_df = pd.DataFrame(
    summary_records
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "step11_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT FINAL SUBSTRUCTURE TABLE
# ============================================================

print(
    "\nTop SHAP bit → substructure summary:"
)

if len(
    substructure_summary_df
) > 0:

    print(
        substructure_summary_df[
            [
                "bit_id",
                "mean_abs_shap",
                "mapped_environment_count",
                "unique_compounds",
                "unique_substructures",
                "most_common_substructure"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No molecular environments were mapped."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "STEP 11 COMPLETE"
)

print(
    "=" * 80
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nGenerated files:"
)


for root, dirs, files in os.walk(
    OUTPUT_DIR
):

    for filename in sorted(files):

        print(
            f"  {os.path.join(root, filename)}"
        )


print(
    "\n" + "=" * 80
)

print(
    "SCIENTIFIC NOTE:"
)

print(
    "Morgan fingerprint bits are hashed structural "
    "features. A single bit can correspond to multiple "
    "molecular environments."
)

print(
    "Therefore, the bit-to-substructure mapping should "
    "be interpreted as an observed molecular-environment "
    "mapping, not as proof that a fingerprint bit "
    "represents one unique chemical functional group."
)

print(
    "=" * 80
)

