# ============================================================
# DIAMOND ANALYTICS PRO
# PRICE PREDICTION UTILITIES
# ============================================================

import numpy as np
import pandas as pd


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NUMERICAL_FEATURES = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "volume",
    "dimension_ratio"
]

CATEGORICAL_FEATURES = [
    "cut",
    "color",
    "clarity"
]

TARGET_COLUMN = "price_inr"


# ============================================================
# VALID CATEGORIES
# ============================================================

CUT_OPTIONS = [
    "Fair",
    "Good",
    "Ideal",
    "Premium",
    "Very Good"
]

COLOR_OPTIONS = [
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
]

CLARITY_OPTIONS = [
    "I1",
    "IF",
    "SI1",
    "SI2",
    "VS1",
    "VS2",
    "VVS1",
    "VVS2"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_volume(x, y, z):
    """
    Calculate diamond volume.
    """

    x = float(x)
    y = float(y)
    z = float(z)

    return x * y * z


def calculate_dimension_ratio(x, y, z):
    """
    Calculate dimension ratio.

    Uses the maximum dimension divided by
    the minimum positive dimension.
    """

    dimensions = [
        float(x),
        float(y),
        float(z)
    ]

    positive_dimensions = [
        value for value in dimensions
        if value > 0
    ]

    if not positive_dimensions:
        return 0.0

    minimum = min(positive_dimensions)
    maximum = max(positive_dimensions)

    if minimum == 0:
        return 0.0

    return maximum / minimum


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe(
    carat,
    depth,
    table,
    x,
    y,
    z,
    cut,
    color,
    clarity
):
    """
    Create one-row dataframe from dashboard inputs.
    """

    volume = calculate_volume(x, y, z)

    dimension_ratio = calculate_dimension_ratio(
        x,
        y,
        z
    )

    data = {
        "carat": float(carat),
        "depth": float(depth),
        "table": float(table),
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "volume": float(volume),
        "dimension_ratio": float(dimension_ratio),
        "cut": str(cut),
        "color": str(color),
        "clarity": str(clarity)
    }

    return pd.DataFrame([data])


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(df):
    """
    Validate prediction input dataframe.
    """

    required_columns = (
        NUMERICAL_FEATURES +
        CATEGORICAL_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return False, (
            f"Missing columns: {missing_columns}"
        )

    # Numerical validation
    for column in NUMERICAL_FEATURES:

        if df[column].isna().any():
            return False, (
                f"Missing value in {column}"
            )

        if not np.isfinite(
            df[column].astype(float)
        ).all():
            return False, (
                f"Invalid numerical value in {column}"
            )

    # Positive physical dimensions
    for column in ["carat", "x", "y", "z"]:

        if (df[column].astype(float) <= 0).any():
            return False, (
                f"{column} must be greater than zero."
            )

    return True, "Input is valid."


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_features(df, encoder):
    """
    Encode categorical features using the
    encoder saved during model training.

    The resulting dataframe is expected to contain
    the same 28 features used by the trained model.
    """

    data = df.copy()

    numerical_data = data[
        NUMERICAL_FEATURES
    ].copy()

    categorical_data = data[
        CATEGORICAL_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Encoder transformation
    # --------------------------------------------------------

    encoded = encoder.transform(
        categorical_data
    )

    # --------------------------------------------------------
    # Obtain encoder feature names
    # --------------------------------------------------------

    try:

        encoded_columns = (
            encoder.get_feature_names_out(
                CATEGORICAL_FEATURES
            )
        )

    except Exception:

        encoded_columns = [
            f"feature_{i}"
            for i in range(
                encoded.shape[1]
            )
        ]

    encoded_data = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=data.index
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_data = pd.concat(
        [
            numerical_data,
            encoded_data
        ],
        axis=1
    )

    return final_data


# ============================================================
# SCALE FEATURES
# ============================================================

def scale_features(df, scaler):
    """
    Apply the saved scaler.

    The scaler was trained on the complete
    encoded feature matrix.
    """

    scaled = scaler.transform(df)

    scaled_df = pd.DataFrame(
        scaled,
        columns=df.columns,
        index=df.index
    )

    return scaled_df


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_prediction_input(
    df,
    encoder,
    scaler
):
    """
    Complete preprocessing pipeline.

    Raw dataframe
        ↓
    Encoding
        ↓
    Scaling
        ↓
    Model-ready dataframe
    """

    valid, message = validate_input(df)

    if not valid:
        raise ValueError(message)

    encoded_data = encode_features(
        df,
        encoder
    )

    scaled_data = scale_features(
        encoded_data,
        scaler
    )

    return scaled_data


# ============================================================
# PREDICT PRICE
# ============================================================

def predict_price(
    df,
    model,
    encoder,
    scaler
):
    """
    Predict diamond price in INR.
    """

    model_input = prepare_prediction_input(
        df,
        encoder,
        scaler
    )

    prediction = model.predict(
        model_input
    )

    price = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return price


# ============================================================
# COMPLETE PREDICTION FUNCTION
# ============================================================

def predict_diamond_price(
    carat,
    depth,
    table,
    x,
    y,
    z,
    cut,
    color,
    clarity,
    model,
    encoder,
    scaler
):
    """
    Dashboard-friendly prediction function.
    """

    input_df = create_input_dataframe(
        carat=carat,
        depth=depth,
        table=table,
        x=x,
        y=y,
        z=z,
        cut=cut,
        color=color,
        clarity=clarity
    )

    price = predict_price(
        input_df,
        model,
        encoder,
        scaler
    )

    return price


# ============================================================
# FORMAT INR
# ============================================================

def format_inr(value):
    """
    Format value as Indian Rupees.
    """

    try:
        return f"₹{float(value):,.2f}"

    except (
        TypeError,
        ValueError
    ):
        return "₹0.00"


# ============================================================
# PRICE CATEGORY
# ============================================================

def get_price_category(price):
    """
    Simple business-friendly price category.
    """

    price = float(price)

    if price < 25000:
        return "Budget"

    elif price < 75000:
        return "Value"

    elif price < 150000:
        return "Premium"

    else:
        return "Luxury"


# ============================================================
# PREDICTION RESULT
# ============================================================

def create_prediction_result(price):
    """
    Return prediction information for dashboard.
    """

    return {
        "price_inr": float(price),
        "formatted_price": format_inr(price),
        "category": get_price_category(price)
    }