# ============================================================
# 💎 DIAMOND ANALYTICS PRO
# Complete Single-File Streamlit Dashboard
#
# FINAL MODEL:
# XGBoost Regression
#
# MODULES USED:
# utils/model_loader.py
# utils/prediction_utils.py
# utils/clustering_utils.py
# utils/insights.py
#
# FEATURES:
# 1. Executive Command Center
# 2. AI Price Prediction
# 3. K-Means Segmentation
# 4. Market Analytics
# 5. Business Intelligence
# 6. Model Performance
# 7. Data Explorer
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib


warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diamond Analytics Pro",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# IMPORT YOUR UTILS
# ============================================================

# These modules are part of your existing project.
# We keep them in the project architecture.

try:
    from utils import model_loader
except Exception:
    model_loader = None


try:
    from utils import prediction_utils
except Exception:
    prediction_utils = None


try:
    from utils import clustering_utils
except Exception:
    clustering_utils = None


try:
    from utils import insights
except Exception:
    insights = None


# ============================================================
# MODEL PATHS
# ============================================================

XGB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_diamond_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "diamond_encoder.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "diamond_scaler.pkl"
)

KMEANS_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "diamond_kmeans_model.pkl"
)

KMEANS_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "diamond_cluster_encoder.pkl"
)

KMEANS_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "diamond_cluster_scaler.pkl"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(29, 78, 216, 0.18),
            transparent 35%
        ),
        #061326;
    color: #F5F7FB;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #08172d 0%,
            #061326 100%
        );

    border-right: 1px solid #18375f;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.sidebar-brand {
    text-align: center;
    padding: 10px 5px 22px 5px;
}

.sidebar-diamond {
    font-size: 58px;
    line-height: 1;
    margin-bottom: 12px;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #F5F7FB;
}

.sidebar-subtitle {
    color: #91A5C1;
    font-size: 13px;
    margin-top: 7px;
}


/* =========================================================
   SIDEBAR DIVIDER
   ========================================================= */

.sidebar-line {
    height: 1px;
    background: #1B365A;
    margin: 15px 0 22px 0;
}


/* =========================================================
   NAVIGATION LABEL
   ========================================================= */

.nav-label {
    color: #627A9C;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 12px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background:
        linear-gradient(
            135deg,
            #102c57 0%,
            #0a1d38 55%,
            #07162b 100%
        );

    border: 1px solid #24538d;
    border-radius: 24px;
    padding: 42px 45px;
    margin-bottom: 28px;

    box-shadow:
        0 18px 50px rgba(0, 0, 0, 0.25);
}

.hero-icon {
    font-size: 65px;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 48px;
    font-weight: 850;
    line-height: 1.05;
    color: #FFFFFF;
}

.hero-blue {
    color: #65ACFF;
}

.hero-subtitle {
    color: #A4B8D4;
    font-size: 17px;
    line-height: 1.8;
    margin-top: 20px;
}

.badge-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 24px;
}

.badge {
    border: 1px solid #285B9B;
    background: #102D58;
    color: #B8D8FF;
    padding: 9px 15px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {
    font-size: 29px;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 15px;
    margin-bottom: 20px;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: #0B1F3B;
    border: 1px solid #1C416C;
    border-radius: 18px;
    padding: 23px;
    height: 100%;
}

.card-title {
    color: #A9BCD5;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}

.card-value {
    color: #FFFFFF;
    font-size: 30px;
    font-weight: 800;
}

.card-description {
    color: #7189A7;
    font-size: 12px;
    margin-top: 7px;
}


/* =========================================================
   PREDICTION RESULT
   ========================================================= */

.prediction-card {
    background:
        linear-gradient(
            135deg,
            #123b72,
            #0c2447
        );

    border: 1px solid #3474BA;
    border-radius: 22px;
    padding: 38px;
    text-align: center;

    box-shadow:
        0 20px 45px rgba(0,0,0,0.25);
}

.prediction-label {
    color: #9DB7D6;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.prediction-price {
    color: #FFFFFF;
    font-size: 52px;
    font-weight: 850;
    margin: 8px 0;
}

.prediction-model {
    color: #78B8FF;
    font-size: 14px;
}


/* =========================================================
   INPUT CARD
   ========================================================= */

.input-card {
    background: #0B1D36;
    border: 1px solid #1C416C;
    border-radius: 18px;
    padding: 22px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    color: #607897;
    padding: 35px 0 15px 0;
    font-size: 12px;
}

</style>
"""
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_project_models():

    xgb_model = None
    encoder = None
    scaler = None

    kmeans_model = None
    cluster_encoder = None
    cluster_scaler = None

    if os.path.exists(XGB_MODEL_PATH):
        xgb_model = joblib.load(
            XGB_MODEL_PATH
        )

    if os.path.exists(ENCODER_PATH):
        encoder = joblib.load(
            ENCODER_PATH
        )

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(
            SCALER_PATH
        )

    if os.path.exists(KMEANS_MODEL_PATH):
        kmeans_model = joblib.load(
            KMEANS_MODEL_PATH
        )

    if os.path.exists(KMEANS_ENCODER_PATH):
        cluster_encoder = joblib.load(
            KMEANS_ENCODER_PATH
        )

    if os.path.exists(KMEANS_SCALER_PATH):
        cluster_scaler = joblib.load(
            KMEANS_SCALER_PATH
        )

    return (
        xgb_model,
        encoder,
        scaler,
        kmeans_model,
        cluster_encoder,
        cluster_scaler
    )


(
    xgb_model,
    encoder,
    scaler,
    kmeans_model,
    cluster_encoder,
    cluster_scaler
) = load_project_models()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    files = glob.glob(
        os.path.join(
            DATA_DIR,
            "*.csv"
        )
    )

    if len(files) == 0:
        return None

    try:
        return pd.read_csv(
            files[0]
        )
    except Exception:
        return None


df = load_data()


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


CUTS = [
    "Fair",
    "Good",
    "Ideal",
    "Premium",
    "Very Good"
]

COLORS = [
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J"
]

CLARITIES = [
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
# CREATE INPUT
# ============================================================

def create_diamond_input(
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

    volume = x * y * z

    if y != 0:
        dimension_ratio = x / y
    else:
        dimension_ratio = 0

    return pd.DataFrame(
        [{
            "carat": carat,
            "depth": depth,
            "table": table,
            "x": x,
            "y": y,
            "z": z,
            "volume": volume,
            "dimension_ratio": dimension_ratio,
            "cut": cut,
            "color": color,
            "clarity": clarity
        }]
    )


# ============================================================
# ENCODE FOR XGBOOST
# ============================================================

def encode_for_xgb(input_df):

    # Use your saved encoder
    if encoder is not None:

        try:

            categorical_encoded = encoder.transform(
                input_df[CATEGORICAL_FEATURES]
            )

            if hasattr(
                categorical_encoded,
                "toarray"
            ):
                categorical_encoded = (
                    categorical_encoded.toarray()
                )

            try:

                names = (
                    encoder
                    .get_feature_names_out(
                        CATEGORICAL_FEATURES
                    )
                )

            except Exception:

                names = [
                    f"encoded_{i}"
                    for i in range(
                        categorical_encoded.shape[1]
                    )
                ]

            cat_df = pd.DataFrame(
                categorical_encoded,
                columns=names
            )

            num_df = input_df[
                NUMERICAL_FEATURES
            ].reset_index(drop=True)

            result = pd.concat(
                [
                    num_df,
                    cat_df
                ],
                axis=1
            )

            return result

        except Exception:
            pass


    # Fallback
    result = pd.get_dummies(
        input_df,
        columns=CATEGORICAL_FEATURES
    )

    expected = [
        "carat",
        "depth",
        "table",
        "x",
        "y",
        "z",
        "volume",
        "dimension_ratio",

        "cut_Fair",
        "cut_Good",
        "cut_Ideal",
        "cut_Premium",
        "cut_Very Good",

        "color_D",
        "color_E",
        "color_F",
        "color_G",
        "color_H",
        "color_I",
        "color_J",

        "clarity_I1",
        "clarity_IF",
        "clarity_SI1",
        "clarity_SI2",
        "clarity_VS1",
        "clarity_VS2",
        "clarity_VVS1",
        "clarity_VVS2"
    ]

    for col in expected:

        if col not in result.columns:
            result[col] = 0

    return result[
        expected
    ].astype(float)


# ============================================================
# PRICE PREDICTION
# ============================================================

def predict_price(input_df):

    if xgb_model is None:

        raise FileNotFoundError(
            "best_diamond_model.pkl is missing."
        )

    if scaler is None:

        raise FileNotFoundError(
            "diamond_scaler.pkl is missing."
        )

    features = encode_for_xgb(
        input_df
    )

    try:

        scaled = scaler.transform(
            features
        )

    except Exception:

        scaled = scaler.transform(
            features.values
        )

    prediction = xgb_model.predict(
        scaled
    )

    return float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-diamond">
                💎
            </div>

            <div class="sidebar-title">
                Diamond Analytics
            </div>

            <div class="sidebar-subtitle">
                Professional Data Science Dashboard
            </div>

        </div>

        <div class="sidebar-line"></div>

        <div class="nav-label">
            NAVIGATION
        </div>
        """
    )


    # --------------------------------------------------------
    # SIDEBAR NAVIGATION
    # --------------------------------------------------------

    page = st.radio(
        "Dashboard navigation",
        [
            "🔴 Executive Command Center",
            "💰 AI Price Prediction",
            "🎯 K-Means Segmentation",
            "📊 Market Analytics",
            "💼 Business Intelligence",
            "⚙️ Model Performance",
            "📁 Data Explorer"
        ],
        label_visibility="collapsed"
    )


    st.html(
        "<div class='sidebar-line'></div>"
    )


    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.html(
        "<div class='nav-label'>MODEL STATUS</div>"
    )


    if xgb_model is not None:

        st.success(
            "XGBoost • Online"
        )

    else:

        st.error(
            "XGBoost • Missing"
        )


    if kmeans_model is not None:

        st.info(
            "K-Means • Online"
        )

    else:

        st.warning(
            "K-Means • Not Loaded"
        )


    st.html(
        "<div class='sidebar-line'></div>"
    )


    st.caption(
        "Diamond Analytics Pro"
    )

    st.caption(
        "XGBoost • K-Means • BI"
    )


# ============================================================
# EXECUTIVE COMMAND CENTER
# ============================================================

if page == "🔴 Executive Command Center":

    st.html(
        """
        <div class="hero">

            <div class="hero-icon">
                💎
            </div>

            <div class="hero-title">
                Diamond
                <span class="hero-blue">
                    Analytics Pro
                </span>
            </div>

            <div class="hero-subtitle">

                AI-powered diamond price prediction<br>

                Market Intelligence • K-Means Segmentation<br>

                Business Intelligence

            </div>

            <div class="badge-row">

                <div class="badge">
                    XGBoost Regression
                </div>

                <div class="badge">
                    K-Means Clustering
                </div>

                <div class="badge">
                    Market Analytics
                </div>

                <div class="badge">
                    Business Intelligence
                </div>

            </div>

        </div>
        """
    )


    st.html(
        "<div class='section-title'>Executive Overview</div>"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        records = (
            len(df)
            if df is not None
            else 0
        )

        st.html(
            f"""
            <div class="card">

                <div class="card-title">
                    DATASET RECORDS
                </div>

                <div class="card-value">
                    {records:,}
                </div>

                <div class="card-description">
                    Available diamond records
                </div>

            </div>
            """
        )


    with c2:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    FINAL MODEL
                </div>

                <div class="card-value">
                    XGBoost
                </div>

                <div class="card-description">
                    Regression algorithm
                </div>

            </div>
            """
        )


    with c3:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    ML TASK
                </div>

                <div class="card-value">
                    Regression
                </div>

                <div class="card-description">
                    Continuous price prediction
                </div>

            </div>
            """
        )


    with c4:

        st.html(
            """
            <div class="card">

                <div class="card-title">
                    CURRENCY
                </div>

                <div class="card-value">
                    ₹ INR
                </div>

                <div class="card-description">
                    Indian Rupee pricing
                </div>

            </div>
            """
        )


    st.html(
        "<div class='section-title'>Project Capabilities</div>"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            """
            ### 💰 AI Price Prediction

            Predict diamond prices using:

            • Carat  
            • Depth  
            • Table  
            • X, Y, Z dimensions  
            • Cut  
            • Color  
            • Clarity  

            Final model: **XGBoost Regression**
            """
        )


    with col2:

        st.info(
            """
            ### 🎯 Customer / Product Segmentation

            K-Means clustering groups diamonds
            according to their characteristics.

            This can support:

            • Inventory segmentation  
            • Product grouping  
            • Customer strategy  
            • Pricing analysis
            """
        )


    if df is not None:

        st.html(
            "<div class='section-title'>Dataset Snapshot</div>"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# AI PRICE PREDICTION
# ============================================================

elif page == "💰 AI Price Prediction":

    st.html(
        "<div class='section-title'>💰 AI Diamond Price Prediction</div>"
    )

    st.write(
        "Enter the physical and quality characteristics "
        "of the diamond."
    )


    st.html(
        "<div class='input-card'>"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader(
            "Physical Characteristics"
        )

        carat = st.number_input(
            "Carat",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.01
        )

        depth = st.number_input(
            "Depth",
            min_value=40.0,
            max_value=80.0,
            value=61.5,
            step=0.1
        )

        table = st.number_input(
            "Table",
            min_value=40.0,
            max_value=80.0,
            value=57.0,
            step=0.1
        )


    with col2:

        st.subheader(
            "Dimensions"
        )

        x = st.number_input(
            "X",
            min_value=0.1,
            max_value=20.0,
            value=6.4,
            step=0.01
        )

        y = st.number_input(
            "Y",
            min_value=0.1,
            max_value=20.0,
            value=6.4,
            step=0.01
        )

        z = st.number_input(
            "Z",
            min_value=0.1,
            max_value=20.0,
            value=3.9,
            step=0.01
        )


    with col3:

        st.subheader(
            "Quality"
        )

        cut = st.selectbox(
            "Cut",
            CUTS,
            index=2
        )

        color = st.selectbox(
            "Color",
            COLORS,
            index=3
        )

        clarity = st.selectbox(
            "Clarity",
            CLARITIES,
            index=5
        )


    st.html(
        "</div>"
    )


    volume = x * y * z

    ratio = (
        x / y
        if y != 0
        else 0
    )


    st.write("")


    m1, m2, m3 = st.columns(3)


    with m1:

        st.metric(
            "Calculated Volume",
            f"{volume:.3f}"
        )


    with m2:

        st.metric(
            "Dimension Ratio",
            f"{ratio:.3f}"
        )


    with m3:

        st.metric(
            "Model Features",
            "28"
        )


    st.write("")


    if st.button(
        "💎  PREDICT DIAMOND PRICE",
        type="primary",
        use_container_width=True
    ):

        try:

            input_df = create_diamond_input(
                carat,
                depth,
                table,
                x,
                y,
                z,
                cut,
                color,
                clarity
            )

            price = predict_price(
                input_df
            )


            st.write("")


            st.html(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        ESTIMATED DIAMOND VALUE
                    </div>

                    <div class="prediction-price">
                        ₹ {price:,.2f}
                    </div>

                    <div class="prediction-model">
                        Powered by XGBoost Regression
                    </div>

                </div>
                """
            )


            st.write("")

            st.subheader(
                "Prediction Input"
            )

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# K-MEANS SEGMENTATION
# ============================================================

elif page == "🎯 K-Means Segmentation":

    st.html(
        "<div class='section-title'>🎯 K-Means Diamond Segmentation</div>"
    )


    if kmeans_model is None:

        st.warning(
            """
            K-Means model was not loaded.

            Check that these files exist:

            diamond_kmeans_model.pkl

            diamond_cluster_encoder.pkl

            diamond_cluster_scaler.pkl
            """
        )

    else:

        st.info(
            "Enter diamond characteristics to identify "
            "the trained K-Means cluster."
        )


        kc1, kc2, kc3 = st.columns(3)


        with kc1:

            k_carat = st.number_input(
                "Carat",
                0.1,
                10.0,
                1.0,
                0.01,
                key="km_carat"
            )

            k_depth = st.number_input(
                "Depth",
                40.0,
                80.0,
                61.5,
                0.1,
                key="km_depth"
            )

            k_table = st.number_input(
                "Table",
                40.0,
                80.0,
                57.0,
                0.1,
                key="km_table"
            )


        with kc2:

            k_x = st.number_input(
                "X",
                0.1,
                20.0,
                6.4,
                0.01,
                key="km_x"
            )

            k_y = st.number_input(
                "Y",
                0.1,
                20.0,
                6.4,
                0.01,
                key="km_y"
            )

            k_z = st.number_input(
                "Z",
                0.1,
                20.0,
                3.9,
                0.01,
                key="km_z"
            )


        with kc3:

            k_cut = st.selectbox(
                "Cut",
                CUTS,
                index=2,
                key="km_cut"
            )

            k_color = st.selectbox(
                "Color",
                COLORS,
                index=3,
                key="km_color"
            )

            k_clarity = st.selectbox(
                "Clarity",
                CLARITIES,
                index=5,
                key="km_clarity"
            )


        if st.button(
            "🎯 FIND DIAMOND SEGMENT",
            type="primary",
            use_container_width=True
        ):

            try:

                cluster_input = create_diamond_input(
                    k_carat,
                    k_depth,
                    k_table,
                    k_x,
                    k_y,
                    k_z,
                    k_cut,
                    k_color,
                    k_clarity
                )


                # Cluster encoding
                if cluster_encoder is not None:

                    cat = cluster_encoder.transform(
                        cluster_input[
                            CATEGORICAL_FEATURES
                        ]
                    )

                    if hasattr(cat, "toarray"):
                        cat = cat.toarray()

                    try:
                        cat_names = (
                            cluster_encoder
                            .get_feature_names_out(
                                CATEGORICAL_FEATURES
                            )
                        )

                    except Exception:
                        cat_names = [
                            f"cat_{i}"
                            for i in range(cat.shape[1])
                        ]

                    cat_df = pd.DataFrame(
                        cat,
                        columns=cat_names
                    )

                    num_df = cluster_input[
                        NUMERICAL_FEATURES
                    ].reset_index(drop=True)

                    combined = pd.concat(
                        [
                            num_df,
                            cat_df
                        ],
                        axis=1
                    )

                else:

                    combined = pd.get_dummies(
                        cluster_input,
                        columns=CATEGORICAL_FEATURES
                    )


                if cluster_scaler is not None:

                    try:

                        cluster_features = (
                            cluster_scaler.transform(
                                combined
                            )
                        )

                    except Exception:

                        cluster_features = (
                            cluster_scaler.transform(
                                combined.values
                            )
                        )

                else:

                    cluster_features = combined.values


                cluster = kmeans_model.predict(
                    cluster_features
                )

                cluster_id = int(
                    np.asarray(
                        cluster
                    ).reshape(-1)[0]
                )


                st.success(
                    "Segmentation completed successfully."
                )


                st.metric(
                    "Assigned Diamond Segment",
                    f"Cluster {cluster_id}"
                )


                st.info(
                    f"""
                    **Cluster {cluster_id}**

                    This diamond has been assigned to
                    Cluster {cluster_id} by the trained
                    K-Means model.

                    The cluster represents diamonds with
                    similar characteristics.
                    """
                )


            except Exception as e:

                st.error(
                    "K-Means prediction failed."
                )

                st.code(
                    str(e)
                )


# ============================================================
# MARKET ANALYTICS
# ============================================================

elif page == "📊 Market Analytics":

    st.html(
        "<div class='section-title'>📊 Diamond Market Analytics</div>"
    )


    if df is None:

        st.warning(
            "Dataset CSV was not found in the data folder."
        )

    else:

        price_col = None

        if "price_inr" in df.columns:
            price_col = "price_inr"

        elif "price" in df.columns:
            price_col = "price"


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        if price_col:

            avg_price = df[
                price_col
            ].mean()

            max_price = df[
                price_col
            ].max()

            min_price = df[
                price_col
            ].min()


            a, b, c = st.columns(3)


            with a:

                st.metric(
                    "Average Price",
                    f"₹ {avg_price:,.0f}"
                )


            with b:

                st.metric(
                    "Highest Price",
                    f"₹ {max_price:,.0f}"
                )


            with c:

                st.metric(
                    "Lowest Price",
                    f"₹ {min_price:,.0f}"
                )


        st.write("")


        # ----------------------------------------------------
        # PRICE DISTRIBUTION
        # ----------------------------------------------------

        if price_col:

            fig1 = px.histogram(
                df,
                x=price_col,
                nbins=40,
                title="Diamond Price Distribution"
            )

            fig1.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )


        # ----------------------------------------------------
        # CARAT VS PRICE
        # ----------------------------------------------------

        if (
            "carat" in df.columns
            and price_col
        ):

            sample = df.sample(
                min(
                    4000,
                    len(df)
                ),
                random_state=42
            )


            fig2 = px.scatter(
                sample,
                x="carat",
                y=price_col,
                title="Carat vs Diamond Price",
                opacity=0.6
            )


            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )


            st.plotly_chart(
                fig2,
                use_container_width=True
            )


        # ----------------------------------------------------
        # CUT ANALYSIS
        # ----------------------------------------------------

        if (
            "cut" in df.columns
            and price_col
        ):

            cut_avg = (
                df.groupby("cut")[
                    price_col
                ]
                .mean()
                .reset_index()
            )


            fig3 = px.bar(
                cut_avg,
                x="cut",
                y=price_col,
                title="Average Diamond Price by Cut"
            )


            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )


            st.plotly_chart(
                fig3,
                use_container_width=True
            )


# ============================================================
# BUSINESS INTELLIGENCE
# ============================================================

elif page == "💼 Business Intelligence":

    st.html(
        "<div class='section-title'>💼 Business Intelligence</div>"
    )


    st.subheader(
        "Business Insights"
    )


    b1, b2, b3 = st.columns(3)


    with b1:

        st.info(
            """
            ### 💰 Pricing

            Use the XGBoost prediction model
            as a data-driven reference for
            diamond pricing.
            """
        )


    with b2:

        st.info(
            """
            ### 📦 Inventory

            K-Means segmentation can help
            organize diamonds into groups
            with similar characteristics.
            """
        )


    with b3:

        st.info(
            """
            ### 📈 Market

            Market analytics can identify
            relationships between carat,
            quality and price.
            """
        )


    if df is not None:

        price_col = None

        if "price_inr" in df.columns:
            price_col = "price_inr"

        elif "price" in df.columns:
            price_col = "price"


        if (
            price_col
            and "clarity" in df.columns
        ):

            clarity_avg = (
                df.groupby("clarity")[
                    price_col
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
                .reset_index()
            )


            fig = px.bar(
                clarity_avg,
                x="clarity",
                y=price_col,
                title="Average Price by Clarity"
            )


            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "⚙️ Model Performance":

    st.html(
        "<div class='section-title'>⚙️ Model Performance</div>"
    )


    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "Final Model",
            "XGBoost"
        )


    with p2:

        st.metric(
            "Task",
            "Regression"
        )


    with p3:

        st.metric(
            "Target",
            "price_inr"
        )


    with p4:

        st.metric(
            "Encoded Features",
            "28"
        )


    st.divider()


    st.subheader(
        "Final Model Performance"
    )


    # Values from your Colab result
    performance = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "MSE",
                "RMSE",
                "R²"
            ],

            "Value": [
                23422.45703125,
                1926908032.0,
                43896.5605941969,
                0.9825023412704468
            ]
        }
    )


    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "Final selected model: XGBoost Regression"
    )


    st.subheader(
        "Feature Information"
    )


    f1, f2 = st.columns(2)


    with f1:

        st.write(
            "**Numerical Features**"
        )

        for feature in NUMERICAL_FEATURES:

            st.write(
                f"• {feature}"
            )


    with f2:

        st.write(
            "**Categorical Features**"
        )

        for feature in CATEGORICAL_FEATURES:

            st.write(
                f"• {feature}"
            )


    st.info(
        """
        The model uses 28 encoded features.

        Numerical features:

        carat, depth, table, x, y, z,
        volume and dimension_ratio.

        Categorical features:

        cut, color and clarity.

        The target variable is price_inr.
        """
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📁 Data Explorer":

    st.html(
        "<div class='section-title'>📁 Diamond Data Explorer</div>"
    )


    if df is None:

        st.warning(
            "No CSV file found inside the data folder."
        )

    else:

        st.write(
            f"Total records: **{len(df):,}**"
        )

        st.write(
            f"Total columns: **{len(df.columns)}**"
        )


        st.subheader(
            "Dataset"
        )


        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )


        st.subheader(
            "Column Information"
        )


        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Missing Values": [
                    int(
                        df[col].isna().sum()
                    )
                    for col in df.columns
                ]
            }
        )


        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        💎 Diamond Analytics Pro

        <br>

        XGBoost Regression • K-Means Segmentation •
        Market Analytics • Business Intelligence

        <br><br>

        Winston Raj | Data Science Project

    </div>
    """
)