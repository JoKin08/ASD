# ============================================================
# 02_check_gnn_dataset.py
# ============================================================

from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd


ROOT = Path(
    r"F:\ASD projoect"
)

CACHE_ROOT = (
    ROOT
    / "GNN_cache_v1"
)

MANIFEST = (
    CACHE_ROOT
    / "cache_manifest.csv"
)


# ============================================================
# EXPECTED CONDITIONS
# ============================================================

EF_CONDITIONS = [
    f"{go}_{context}_{emotion}"
    for go, context, emotion
    in product(
        [
            "go",
            "nogo"
        ],
        [
            "rf",
            "cf"
        ],
        [
            "happy",
            "angry",
            "surprise",
            "neutral"
        ]
    )
]


TOM_CONDITIONS = [
    f"{belief}_{order}_{tom}"
    for belief, order, tom
    in product(
        [
            "true",
            "false"
        ],
        [
            "first",
            "second"
        ],
        [
            "cognitive",
            "affective"
        ]
    )
]


EXPECTED = {
    "EF": EF_CONDITIONS,
    "ToM_H": TOM_CONDITIONS,
    "ToM_T": TOM_CONDITIONS,
}


def main():

    df = pd.read_csv(
        MANIFEST
    )

    print(
        "\nManifest rows:",
        len(df)
    )

    print(
        "\nModels:",
        df["model"].unique()
    )

    # ========================================================
    # SUBJECT COUNTS
    # ========================================================

    print(
        "\n========================="
    )
    print(
        "SUBJECT COUNTS"
    )
    print(
        "========================="
    )

    counts = (
        df.groupby(
            [
                "model",
                "group"
            ]
        )[
            "subject_id"
        ]
        .nunique()
    )

    print(
        counts
    )

    # ========================================================
    # DUPLICATE SUBJECT-CONDITION CACHE
    # ========================================================

    duplicates = (
        df.groupby(
            [
                "model",
                "group",
                "subject_id",
                "condition"
            ]
        )
        .size()
    )

    duplicates = duplicates[
        duplicates > 1
    ]

    print(
        "\nDuplicate "
        "subject-condition entries:"
    )

    if len(duplicates) == 0:
        print(
            "None"
        )
    else:
        print(
            duplicates
        )

    # ========================================================
    # CONDITION COVERAGE
    # ========================================================

    for model in [
        "EF",
        "ToM_H",
        "ToM_T"
    ]:

        print(
            "\n\n========================="
        )
        print(
            model
        )
        print(
            "========================="
        )

        model_df = df[
            df["model"] == model
        ]

        if len(model_df) == 0:
            print(
                "No data."
            )
            continue

        expected = set(
            EXPECTED[model]
        )

        found = set(
            model_df[
                "condition"
            ].unique()
        )

        print(
            "\nExpected conditions:",
            len(expected)
        )

        print(
            "Found conditions:",
            len(found)
        )

        missing_global = (
            expected - found
        )

        unexpected = (
            found - expected
        )

        if missing_global:
            print(
                "\nMissing globally:"
            )

            for x in sorted(
                missing_global
            ):
                print(
                    " ",
                    x
                )

        if unexpected:
            print(
                "\nUnexpected labels:"
            )

            for x in sorted(
                unexpected
            ):
                print(
                    " ",
                    x
                )

        # ====================================================
        # CONDITION STATISTICS
        # ====================================================

        summary = (
            model_df
            .groupby(
                [
                    "group",
                    "condition"
                ]
            )
            .agg(
                n_subjects=(
                    "subject_id",
                    "nunique"
                ),
                min_trials=(
                    "n_trials",
                    "min"
                ),
                median_trials=(
                    "n_trials",
                    "median"
                ),
                mean_trials=(
                    "n_trials",
                    "mean"
                ),
                max_trials=(
                    "n_trials",
                    "max"
                ),
            )
            .reset_index()
        )

        print(
            "\nCondition coverage:"
        )

        print(
            summary.to_string(
                index=False
            )
        )

        # ====================================================
        # MISSING CONDITIONS PER SUBJECT
        # ====================================================

        print(
            "\nSubjects missing "
            "one or more conditions:"
        )

        n_missing_subjects = 0

        for (
            group,
            subject_id
        ), subject_df in (
            model_df.groupby(
                [
                    "group",
                    "subject_id"
                ]
            )
        ):

            subject_conditions = set(
                subject_df[
                    "condition"
                ]
            )

            missing = (
                expected
                - subject_conditions
            )

            if missing:

                n_missing_subjects += 1

                print(
                    f"{group} "
                    f"{subject_id}: "
                    f"missing "
                    f"{len(missing)}"
                )

                print(
                    "   ",
                    sorted(missing)
                )

        print(
            "\nNumber with incomplete "
            f"condition coverage: "
            f"{n_missing_subjects}"
        )

        # ====================================================
        # VERY LOW TRIAL COUNTS
        # ====================================================

        low = model_df[
            model_df[
                "n_trials"
            ] < 10
        ]

        print(
            "\nSubject-condition cells "
            "with <10 trials:"
        )

        if len(low) == 0:
            print(
                "None"
            )
        else:

            print(
                low[
                    [
                        "group",
                        "subject_id",
                        "condition",
                        "n_trials"
                    ]
                ]
                .sort_values(
                    "n_trials"
                )
                .to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()