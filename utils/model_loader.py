# ============================================================
# DIAMOND ANALYTICS PRO
# MODEL LOADER
# ============================================================

from pathlib import Path
import joblib


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------
# MODEL FILES
# ------------------------------------------------------------

BEST_MODEL_FILE = MODELS_DIR / "best_diamond_model.pkl"

ENCODER_FILE = MODELS_DIR / "diamond_encoder.pkl"

SCALER_FILE = MODELS_DIR / "diamond_scaler.pkl"

FEATURE_INFO_FILE = MODELS_DIR / "diamond_feature_information.pkl"

KMEANS_MODEL_FILE = MODELS_DIR / "diamond_kmeans_model.pkl"

CLUSTER_ENCODER_FILE = MODELS_DIR / "diamond_cluster_encoder.pkl"

CLUSTER_SCALER_FILE = MODELS_DIR / "diamond_cluster_scaler.pkl"


# ------------------------------------------------------------
# GENERIC LOADER
# ------------------------------------------------------------

def load_pickle(file_path):
    """
    Load a pickle/joblib model or object.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required model file was not found:\n{file_path}"
        )

    return joblib.load(file_path)


# ------------------------------------------------------------
# LOAD FINAL PRICE MODEL
# ------------------------------------------------------------

def load_best_model():
    """
    Load the final XGBoost regression model.
    """

    return load_pickle(BEST_MODEL_FILE)


# ------------------------------------------------------------
# LOAD PRICE ENCODER
# ------------------------------------------------------------

def load_price_encoder():
    """
    Load categorical encoder used for price prediction.
    """

    return load_pickle(ENCODER_FILE)


# ------------------------------------------------------------
# LOAD PRICE SCALER
# ------------------------------------------------------------

def load_price_scaler():
    """
    Load numerical feature scaler used during training.
    """

    return load_pickle(SCALER_FILE)


# ------------------------------------------------------------
# LOAD FEATURE INFORMATION
# ------------------------------------------------------------

def load_feature_information():
    """
    Load saved feature information if available.
    """

    if not FEATURE_INFO_FILE.exists():
        return None

    return load_pickle(FEATURE_INFO_FILE)


# ------------------------------------------------------------
# LOAD CLUSTERING MODEL
# ------------------------------------------------------------

def load_kmeans_model():
    """
    Load K-Means clustering model.
    """

    return load_pickle(KMEANS_MODEL_FILE)


# ------------------------------------------------------------
# LOAD CLUSTER ENCODER
# ------------------------------------------------------------

def load_cluster_encoder():
    """
    Load encoder used for clustering.
    """

    return load_pickle(CLUSTER_ENCODER_FILE)


# ------------------------------------------------------------
# LOAD CLUSTER SCALER
# ------------------------------------------------------------

def load_cluster_scaler():
    """
    Load scaler used for clustering.
    """

    return load_pickle(CLUSTER_SCALER_FILE)


# ------------------------------------------------------------
# LOAD ALL PRICE COMPONENTS
# ------------------------------------------------------------

def load_prediction_components():
    """
    Load all components required for price prediction.
    """

    model = load_best_model()
    encoder = load_price_encoder()
    scaler = load_price_scaler()
    feature_information = load_feature_information()

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "feature_information": feature_information
    }


# ------------------------------------------------------------
# CHECK AVAILABLE MODELS
# ------------------------------------------------------------

def check_model_files():
    """
    Return availability status of model files.
    """

    files = {
        "best_model": BEST_MODEL_FILE,
        "encoder": ENCODER_FILE,
        "scaler": SCALER_FILE,
        "feature_information": FEATURE_INFO_FILE,
        "kmeans_model": KMEANS_MODEL_FILE,
        "cluster_encoder": CLUSTER_ENCODER_FILE,
        "cluster_scaler": CLUSTER_SCALER_FILE
    }

    return {
        name: path.exists()
        for name, path in files.items()
    }