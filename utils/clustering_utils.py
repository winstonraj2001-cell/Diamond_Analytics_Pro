# ============================================================
# DIAMOND ANALYTICS PRO
# K-MEANS CLUSTERING UTILITIES
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# CLUSTER FEATURES
# ============================================================

CLUSTER_NUMERICAL_FEATURES = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z"
]

CLUSTER_CATEGORICAL_FEATURES = [
    "cut",
    "color",
    "clarity"
]


# ============================================================
# CREATE CLUSTER INPUT
# ============================================================

def create_cluster_dataframe(df):
    """
    Create dataframe containing clustering features.
    """

    data = df.copy()

    required = (
        CLUSTER_NUMERICAL_FEATURES +
        CLUSTER_CATEGORICAL_FEATURES
    )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing clustering columns: {missing}"
        )

    return data[required].copy()


# ============================================================
# PREPARE CLUSTER INPUT
# ============================================================

def prepare_cluster_input(
    df,
    encoder,
    scaler
):
    """
    Prepare dataframe for K-Means.
    """

    data = create_cluster_dataframe(df)

    numerical = data[
        CLUSTER_NUMERICAL_FEATURES
    ].copy()

    categorical = data[
        CLUSTER_CATEGORICAL_FEATURES
    ].copy()

    encoded = encoder.transform(
        categorical
    )

    try:

        encoded_columns = (
            encoder.get_feature_names_out(
                CLUSTER_CATEGORICAL_FEATURES
            )
        )

    except Exception:

        encoded_columns = [
            f"category_{i}"
            for i in range(
                encoded.shape[1]
            )
        ]

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=data.index
    )

    combined = pd.concat(
        [
            numerical,
            encoded_df
        ],
        axis=1
    )

    scaled = scaler.transform(
        combined
    )

    return scaled


# ============================================================
# PREDICT CLUSTER
# ============================================================

def predict_cluster(
    df,
    kmeans_model,
    encoder,
    scaler
):
    """
    Predict K-Means cluster.
    """

    cluster_input = prepare_cluster_input(
        df,
        encoder,
        scaler
    )

    labels = kmeans_model.predict(
        cluster_input
    )

    return labels


# ============================================================
# GET SINGLE CLUSTER
# ============================================================

def get_single_cluster(
    df,
    kmeans_model,
    encoder,
    scaler
):
    """
    Return cluster for one diamond.
    """

    labels = predict_cluster(
        df,
        kmeans_model,
        encoder,
        scaler
    )

    return int(
        np.asarray(labels).reshape(-1)[0]
    )


# ============================================================
# CLUSTER SUMMARY
# ============================================================

def create_cluster_summary(
    df,
    cluster_column="cluster"
):
    """
    Create summary statistics for clusters.
    """

    if cluster_column not in df.columns:
        raise ValueError(
            f"{cluster_column} not found."
        )

    numeric_columns = [
        column
        for column in [
            "carat",
            "price",
            "price_inr"
        ]
        if column in df.columns
    ]

    if not numeric_columns:
        return (
            df[cluster_column]
            .value_counts()
            .reset_index()
        )

    summary = (
        df.groupby(cluster_column)
        .agg(
            diamond_count=(
                cluster_column,
                "size"
            ),
            **{
                f"average_{column}": (
                    column,
                    "mean"
                )
                for column in numeric_columns
            }
        )
        .reset_index()
    )

    return summary


# ============================================================
# CLUSTER LABEL
# ============================================================

def get_cluster_name(cluster_id):
    """
    Convert cluster number into
    dashboard-friendly market segment.
    """

    labels = {
        0: "Value Segment",
        1: "Premium Segment",
        2: "Luxury Segment",
        3: "Budget Segment",
        4: "Emerging Segment"
    }

    return labels.get(
        int(cluster_id),
        f"Segment {cluster_id}"
    )