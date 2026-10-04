"""
This file documents the final model-building step.

Recommended final workflow:

1. Train the FINAL XGBoost model using the exact same preprocessing used
   during evaluation.
2. Save it as model.json.
3. Convert model.json to model.onnx using onnxmltools.
4. Put model.onnx in ../public/model.onnx.
5. Put the final hyperparameters and feature order in
   ../public/model_metadata.json.

The current project assumes this exact 19-feature order:

0  Built-up Area (m2)
1  Number of Floors (count)
2  Building Height (m)
3  Service Life (years)
4  Cement (tonnes)
5  Steel Reinforcement (tonnes)
6  Bricks/Blocks (tonnes)
7  Sand (tonnes)
8  Aggregate (tonnes)
9  Glass (tonnes)
10 Wood (m3)
11 Transportation Distance (km)
12 Construction Energy (kWh)
13 Annual Electricity Consumption (kWh/year)
14 Building Type_Commercial
15 Building Type_Industrial
16 Building Type_Institutional
17 Building Type_Office
18 Building Type_Residential

IMPORTANT:
If your actual training code uses a different one-hot order, different
features, or additional preprocessing, change src/predictor.js to match it
exactly. The browser model must receive the same feature representation as
the Python model.
"""
