import { useState } from "react";
import { predictGWP } from "./predictor";

const initialForm = {
  buildingType: "Residential",
  builtUpArea: 2000,
  floors: 3,
  height: 10,
  serviceLife: 50,
  cement: 300,
  steel: 50,
  bricks: 100,
  sand: 150,
  aggregate: 250,
  glass: 10,
  wood: 5,
  transportDistance: 25,
  constructionEnergy: 5000,
  annualElectricity: 25000
};

function Field({ label, unit, value, onChange, type = "number", step = "any", min = "0" }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <div className="input-wrap">
        <input
          type={type}
          min={type === "number" ? min : undefined}
          step={type === "number" ? step : undefined}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        {unit && <span className="unit">{unit}</span>}
      </div>
    </label>
  );
}

function Section({ title, children }) {
  return (
    <section className="form-section">
      <h2>{title}</h2>
      <div className="field-grid">{children}</div>
    </section>
  );
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const update = (key) => (value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  async function handlePredict(e) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const prediction = await predictGWP(form);
      setResult(prediction);
    } catch (err) {
      console.error(err);
      setError(
        "The XGBoost model is not loaded yet. Add public/model.onnx generated from your final trained model, then run the app again."
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setForm(initialForm);
    setResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="navbar">
        <div className="brand">
          <div className="brand-mark">G</div>
          <div>
            <div className="brand-name">GreenBuild <span>AI</span></div>
            <div className="brand-subtitle">Life-cycle GWP Prediction</div>
          </div>
        </div>

        <nav>
          <a className="active" href="#dashboard">Dashboard</a>
          <a href="#model">Dataset &amp; Model</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <main id="dashboard" className="container">
        <div className="hero">
          <div>
            <p className="eyebrow">BUILDING CARBON ANALYSIS</p>
            <h1>Predict your building's<br /><span>Global Warming Potential</span></h1>
            <p className="hero-copy">
              Enter the building and material parameters below. The trained
              XGBoost model predicts total life-cycle GWP directly in your browser.
            </p>
          </div>
          <div className="model-badge">
            <div className="dot" />
            XGBoost Model
          </div>
        </div>

        <div className="dashboard-grid">
          <form className="card form-card" onSubmit={handlePredict}>
            <Section title="Building Information">
              <label className="field">
                <span className="field-label">Building Type</span>
                <select
                  value={form.buildingType}
                  onChange={(e) => update("buildingType")(e.target.value)}
                >
                  <option>Residential</option>
                  <option>Commercial</option>
                  <option>Industrial</option>
                  <option>Institutional</option>
                  <option>Office</option>
                </select>
              </label>

              <Field label="Built-up Area" unit="m²" value={form.builtUpArea} onChange={update("builtUpArea")} />
              <Field label="Number of Floors" unit="count" value={form.floors} onChange={update("floors")} step="1" />
              <Field label="Building Height" unit="m" value={form.height} onChange={update("height")} />
              <Field label="Service Life" unit="years" value={form.serviceLife} onChange={update("serviceLife")} step="1" />
            </Section>

            <Section title="Material Quantities">
              <Field label="Cement" unit="tonnes" value={form.cement} onChange={update("cement")} />
              <Field label="Steel Reinforcement" unit="tonnes" value={form.steel} onChange={update("steel")} />
              <Field label="Bricks / Blocks" unit="tonnes" value={form.bricks} onChange={update("bricks")} />
              <Field label="Sand" unit="tonnes" value={form.sand} onChange={update("sand")} />
              <Field label="Aggregate" unit="tonnes" value={form.aggregate} onChange={update("aggregate")} />
              <Field label="Glass" unit="tonnes" value={form.glass} onChange={update("glass")} />
              <Field label="Wood" unit="m³" value={form.wood} onChange={update("wood")} />
            </Section>

            <Section title="Energy & Transportation">
              <Field label="Transportation Distance" unit="km" value={form.transportDistance} onChange={update("transportDistance")} />
              <Field label="Construction Energy" unit="kWh" value={form.constructionEnergy} onChange={update("constructionEnergy")} />
              <Field label="Annual Electricity Consumption" unit="kWh/year" value={form.annualElectricity} onChange={update("annualElectricity")} />
            </Section>

            <div className="actions">
              <button className="primary" type="submit" disabled={loading}>
                {loading ? "Predicting..." : "Predict GWP →"}
              </button>
              <button className="secondary" type="button" onClick={reset}>
                Reset
              </button>
            </div>
          </form>

          <aside className="side-column">
            <div className="card result-card">
              <div className="card-heading">
                <div>
                  <p className="eyebrow">PREDICTION</p>
                  <h2>Life-cycle GWP</h2>
                </div>
                <div className="result-icon">CO₂</div>
              </div>

              {result === null ? (
                <div className="empty-result">
                  <div className="empty-number">—</div>
                  <p>Enter your building parameters and run the XGBoost model.</p>
                </div>
              ) : (
                <div className="prediction">
                  <div className="prediction-number">
                    {result.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </div>
                  <div className="prediction-unit">kg CO₂-eq</div>
                  <div className="per-area">
                    {(
                      result / Number(form.builtUpArea)
                    ).toLocaleString(undefined, { maximumFractionDigits: 2 })} kg CO₂-eq/m²
                  </div>
                </div>
              )}

              {error && <div className="error-box">{error}</div>}
            </div>

            <div id="model" className="card model-card">
              <p className="eyebrow">MODEL INFORMATION</p>
              <h2>XGBoost Regressor</h2>
              <div className="model-row"><span>Target</span><strong>Total life-cycle GWP</strong></div>
              <div className="model-row"><span>Inference</span><strong>In-browser</strong></div>
              <div className="model-row"><span>Model format</span><strong>ONNX</strong></div>
              <div className="model-row"><span>Estimators</span><strong>300</strong></div>
              <div className="model-row"><span>Max depth</span><strong>3</strong></div>
              <div className="model-row"><span>Learning rate</span><strong>0.05</strong></div>
              <div className="model-row"><span>Encoding</span><strong>One-hot / drop first</strong></div>

              <div className="model-note">
                Configuration matches the final XGBoost training code.
              </div>
            </div>
          </aside>
        </div>

        <section id="about" className="methodology">
          <div>
            <p className="eyebrow">METHODOLOGY</p>
            <h2>What the prediction represents</h2>
            <p>
              The target is total life-cycle global warming potential:
              A1–A3 + A4 + A5 + B6, expressed as kg CO₂-equivalent.
              The deployed model learns the relationship between the building
              parameters and this target.
            </p>
          </div>
          <div className="formula">
            <span>GWP</span>
            <b>=</b>
            <span>A1–A3</span>
            <b>+</b>
            <span>A4</span>
            <b>+</b>
            <span>A5</span>
            <b>+</b>
            <span>B6</span>
          </div>
        </section>

        <footer>
          GreenBuild AI · Academic Research Prototype
        </footer>
      </main>
    </div>
  );
}

export default App;
