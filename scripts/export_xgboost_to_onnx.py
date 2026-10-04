"""
Convert a trained XGBoost regression model to ONNX.

Install first:
    pip install xgboost onnxmltools onnx numpy

Usage:
    python scripts/export_xgboost_to_onnx.py model.json public/model.onnx 19

If your model was saved from an XGBRegressor:
    model.save_model("model.json")

This script assumes the model accepts a dense float32 matrix.
"""

import sys
import numpy as np
import xgboost as xgb

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/export_xgboost_to_onnx.py MODEL_JSON OUTPUT_ONNX N_FEATURES")
        raise SystemExit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    n_features = int(sys.argv[3])

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    initial_types = [
        ("input", FloatTensorType([None, n_features]))
    ]

    onnx_model = convert_xgboost(model, initial_types=initial_types)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
