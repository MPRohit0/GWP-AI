# GreenBuild AI

Static React + Vite dashboard for predicting total life-cycle building GWP
using the exact XGBoost configuration from the research notebook.

## Final XGBoost model

```python
XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

Other XGBoost parameters use their defaults.

The research notebook uses:

```python
data = data.drop(["Building ID"], axis=1)

data = pd.get_dummies(
    data,
    columns=["Building Type"],
    drop_first=True,
    dtype=int
)
```

Therefore `Commercial` is the reference category and the model receives
four building-type columns:

```text
Building Type_Industrial
Building Type_Institutional
Building Type_Office
Building Type_Residential
```

Total input features = **18**.

## Run locally

Install Node.js 20+.

Then:

```bash
npm install
npm run dev
```

Open the URL printed by Vite.

## Create the real browser model

The website needs:

```text
public/model.onnx
```

The ONNX model must be the **same final XGBoost model** used for your research.

### Option A — save the model from your notebook

After your existing XGBoost definition, fit it on the complete dataset:

```python
xgb_final = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

xgb_final.fit(X, y)

xgb_final.save_model("model.json")
```

Install conversion packages:

```bash
pip install xgboost onnxmltools onnx numpy
```

Then:

```bash
python scripts/export_xgboost_to_onnx.py model.json public/model.onnx 18
```

### Important research distinction

Your 4-fold CV score is the validation result:

```text
Fold 1: 0.52544790
Fold 2: 0.91742092
Fold 3: 0.92195397
Fold 4: 0.95787058

Mean R²: 0.83067334
```

The deployed website should use a model trained on the **full 50-row dataset**
after the CV evaluation is finished. Do not use a fold-specific model for the
website.

## Exact browser feature order

The browser sends:

```text
1.  Built-up Area (m2)
2.  Number of Floors (count)
3.  Building Height (m)
4.  Service Life (years)
5.  Cement (tonnes)
6.  Steel Reinforcement (tonnes)
7.  Bricks/Blocks (tonnes)
8.  Sand (tonnes)
9.  Aggregate (tonnes)
10. Glass (tonnes)
11. Wood (m3)
12. Transportation Distance (km)
13. Construction Energy (kWh)
14. Annual Electricity Consumption (kWh/year)
15. Building Type_Industrial
16. Building Type_Institutional
17. Building Type_Office
18. Building Type_Residential
```

This order MUST match the final `X.columns` from Python.

## GitHub Pages

The Vite configuration currently uses:

```js
base: "/greenbuild-ai/"
```

This assumes the GitHub repository is named:

```text
greenbuild-ai
```

If your repository has another name, change the base accordingly.

Push the project to GitHub. The included GitHub Actions workflow will build
and deploy it.

In GitHub:

```text
Repository
→ Settings
→ Pages
→ Source: GitHub Actions
```

## Research note

The target is:

```text
GWP = A1-A3 + A4 + A5 + B6
```

in kg CO2-equivalent.

Because the target was constructed from building/material parameters using LCA
equations, the ML model is learning a generated target from those same inputs.
This needs to be stated clearly in the research paper when interpreting the
R² values.
