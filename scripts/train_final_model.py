"""
Train the FINAL XGBoost model exactly as in the research notebook and export
it to ONNX for the GreenBuild AI GitHub Pages website.

Run from the project root:

    pip install pandas numpy xgboost onnxmltools onnx scikit-learn
    python scripts/train_final_model.py

Input:
    building_data_50_model_ready.csv

Outputs:
    public/model.json
    public/model.onnx

IMPORTANT:
The website uses the full-data final model. The R2 numbers from the notebook
are 4-fold cross-validation results and are not the model being deployed.

Why NumPy is used for the final fit:
Some onnxmltools versions cannot convert XGBoost models whose internal
feature names are pandas column names such as "Sand (tonnes)". They expect
feature names like f0, f1, f2, etc. Converting X to NumPy before fitting
preserves the exact column ORDER while making the model compatible with the
ONNX converter.
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "building_data_50_model_ready.csv"
MODEL_JSON = ROOT / "public" / "model.json"
MODEL_ONNX = ROOT / "public" / "model.onnx"


def main():
    # ------------------------------------------------------------
    # 1. Load exactly the same dataset as the research notebook
    # ------------------------------------------------------------
    data = pd.read_csv(DATA_PATH)

    # Same as your notebook
    data = data.drop(["Building ID"], axis=1)

    # Same categorical encoding as your notebook
    data = pd.get_dummies(
        data,
        columns=["Building Type"],
        drop_first=True,
        dtype=int
    )

    # ------------------------------------------------------------
    # 2. Separate X and y
    # ------------------------------------------------------------
    X = data.drop(["y_Total_Life_Cycle_GWP_kgCO2eq"], axis=1)
    y = data["y_Total_Life_Cycle_GWP_kgCO2eq"]

    # This is the exact order expected by the website.
    expected_features = [
        "Built-up Area (m2)",
        "Number of Floors (count)",
        "Building Height (m)",
        "Service Life (years)",
        "Cement (tonnes)",
        "Steel Reinforcement (tonnes)",
        "Bricks/Blocks (tonnes)",
        "Sand (tonnes)",
        "Aggregate (tonnes)",
        "Glass (tonnes)",
        "Wood (m3)",
        "Transportation Distance (km)",
        "Construction Energy (kWh)",
        "Annual Electricity Consumption (kWh/year)",
        "Building Type_Industrial",
        "Building Type_Institutional",
        "Building Type_Office",
        "Building Type_Residential",
    ]

    if list(X.columns) != expected_features:
        raise ValueError(
            "\nFeature order does not match the website.\n\n"
            f"Actual:\n{list(X.columns)}\n\n"
            f"Expected:\n{expected_features}\n"
        )

    # ------------------------------------------------------------
    # 3. FINAL XGBoost model
    #    EXACTLY your research parameters
    # ------------------------------------------------------------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    # IMPORTANT:
    # Convert to NumPy before fitting.
    #
    # The order is unchanged:
    # column 0 -> f0
    # column 1 -> f1
    # ...
    # column 17 -> f17
    #
    # This avoids the onnxmltools feature-name conversion bug.
    X_numpy = X.to_numpy(dtype="float32")
    y_numpy = y.to_numpy(dtype="float32")

    model.fit(X_numpy, y_numpy)

    # ------------------------------------------------------------
    # 4. Save native XGBoost model
    # ------------------------------------------------------------
    MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_JSON)

    # ------------------------------------------------------------
    # 5. Convert XGBoost -> ONNX
    # ------------------------------------------------------------
    initial_types = [
        ("input", FloatTensorType([None, len(expected_features)]))
    ]

    onnx_model = convert_xgboost(
        model,
        initial_types=initial_types
    )

    MODEL_ONNX.write_bytes(
        onnx_model.SerializeToString()
    )

    print("\nSUCCESS!")
    print("----------------------------------------")
    print("Final model trained on all 50 records.")
    print(f"Number of features: {len(expected_features)}")
    print("")
    print("XGBoost parameters:")
    print("  n_estimators = 300")
    print("  max_depth    = 3")
    print("  learning_rate = 0.05")
    print("  random_state = 42")
    print("")
    print(f"XGBoost model: {MODEL_JSON}")
    print(f"ONNX model:    {MODEL_ONNX}")
    print("----------------------------------------")
    print("\nThe ONNX model is now ready for the website.")


if __name__ == "__main__":
    main()
