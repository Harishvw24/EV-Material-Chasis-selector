import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:5000";

const defaultValues = {
  Su: "450",
  Sy: "275",
  E: "198000",
  G: "77000",
  mu: "0.29",
  Ro: "7820",
};

const fieldMeta = [
  { key: "Su", label: "Ultimate Strength (Su)" },
  { key: "Sy", label: "Yield Strength (Sy)" },
  { key: "E", label: "Elastic Modulus (E)" },
  { key: "G", label: "Shear Modulus (G)" },
  { key: "mu", label: "Poisson Ratio (mu)" },
  { key: "Ro", label: "Density (Ro)" },
];

function App() {
  const [inputs, setInputs] = useState(defaultValues);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const inputList = useMemo(() => fieldMeta, []);

  const handleChange = (key, value) => {
    setInputs((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setStatus("loading");
    setError("");
    setResult(null);

    try {
      const payload = Object.fromEntries(
        Object.entries(inputs).map(([key, value]) => [key, Number(value)])
      );
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "Prediction failed.");
      }
      setResult(data);
      setStatus("success");
    } catch (err) {
      setError(err.message || "Request failed.");
      setStatus("error");
    }
  };

  const handleReset = () => {
    setInputs(defaultValues);
    setResult(null);
    setError("");
    setStatus("idle");
  };

  return (
    <div className="app-shell">
      <main className="card">
        <div className="card__header">
          <p className="eyebrow">Material Vhasis</p>
          <h1>Material selection predictor</h1>
          <p className="subtitle">
            Enter mechanical properties to check if the material is suitable
            for selection.
          </p>
        </div>

        <form className="grid" onSubmit={handleSubmit}>
          {inputList.map((field) => (
            <label className="field" key={field.key}>
              <span>{field.label}</span>
              <input
                type="number"
                step="any"
                value={inputs[field.key]}
                onChange={(event) => handleChange(field.key, event.target.value)}
                required
              />
            </label>
          ))}

          <div className="actions">
            <button type="submit" disabled={status === "loading"}>
              {status === "loading" ? "Predicting..." : "Run prediction"}
            </button>
            <button type="button" className="ghost" onClick={handleReset}>
              Reset
            </button>
          </div>
        </form>

        {status === "error" && <p className="message error">{error}</p>}

        {result && (
          <div className="result">
            <div>
              <p className="result__label">Prediction</p>
              <p className="result__value">
                {result.usable ? "Suitable" : "Not suitable"}
              </p>
            </div>
            <div>
              <p className="result__label">Probability</p>
              <p className="result__value">
                {result.probability
                  ? `${(result.probability * 100).toFixed(1)}%`
                  : "N/A"}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
