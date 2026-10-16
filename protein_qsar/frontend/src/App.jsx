import { useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "https://qsar-x.onrender.com";
const HISTORY_KEY = "qsar_x_prediction_history";
const exampleSmiles = "CCOc1ccc2nc(S(N)(=O)=O)sc2c1";

function App() {
  const [smiles, setSmiles] = useState("");
  const [targets, setTargets] = useState([]);
  const [catalogTargets, setCatalogTargets] = useState([]);
  const [selectedTargetId, setSelectedTargetId] = useState("");
  const [targetQuery, setTargetQuery] = useState("");
  const [targetMessage, setTargetMessage] = useState("");
  const [targetResolving, setTargetResolving] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [batchFile, setBatchFile] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [batchError, setBatchError] = useState("");
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
  });

  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); }, [history]);

  useEffect(() => {
    Promise.all([
      fetch(`${API_URL}/health`),
      fetch(`${API_URL}/targets`),
      fetch(`${API_URL}/research-targets`),
    ])
      .then(async ([healthResponse, targetsResponse, catalogTargetsResponse]) => {
        if (!healthResponse.ok || !targetsResponse.ok || !catalogTargetsResponse.ok) {
          throw new Error("QSAR API is unavailable");
        }
        const [data, catalogData] = await Promise.all([targetsResponse.json(), catalogTargetsResponse.json()]);
        setTargets(data.targets || []);
        setCatalogTargets(catalogData.targets || []);
        setApiConnected(true);
      })
      .catch(() => {
        setTargets([]);
        setCatalogTargets([]);
        setApiConnected(false);
      });
  }, []);

  const selectedTarget = targets.find(
    (target) => target.target_id === selectedTargetId
  );
  const targetOptions = [...targets, ...catalogTargets].filter(
    (target, index, all) => all.findIndex((item) => item.target_id === target.target_id) === index
  );
  const selectedCatalogTarget = targetOptions.find((target) => target.target_id === selectedTargetId);
  const canPredict = Boolean(selectedTarget?.prediction_available);

  const resolveProteinTarget = async () => {
    if (!targetQuery.trim()) {
      setTargetMessage("Enter a protein name, gene, target ID, UniProt accession, or protein sequence.");
      setSelectedTargetId("");
      return;
    }
    setTargetResolving(true);
    setTargetMessage("");
    setSelectedTargetId("");
    try {
      const response = await fetch(`${API_URL}/resolve-target`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ protein_target: targetQuery.trim() }),
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = data.detail;
        throw new Error((typeof detail === "string" ? detail : detail?.message) || "Target could not be resolved.");
      }
      setSelectedTargetId(data.target.target_id);
      setTargetMessage(
        data.prediction_available
          ? `Resolved ${data.target.gene} · ${data.target.target_id} · Target-specific model available`
          : `Resolved ${data.target.gene} · ${data.target.target_id} · Target-specific model training/validation is in progress`
      );
      setResult(null);
    } catch (resolveError) {
      setTargetMessage(resolveError.message);
    } finally {
      setTargetResolving(false);
    }
  };

  // =========================
  // SINGLE PREDICTION
  // =========================

  const predictActivity = async () => {
    if (!smiles.trim()) {
      setError("Please enter a SMILES structure.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          smiles: smiles.trim(),
          target_id: selectedTargetId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        const detail = data.detail;
        throw new Error(
          (typeof detail === "string" ? detail : detail?.message) ||
            "Prediction failed."
        );
      }

      const prediction = data.result;

      setResult(prediction);
      setHistory((previous) => [{
        id: Date.now(), createdAt: new Date().toLocaleString(), target_id: selectedTargetId,
        target_gene: prediction.target?.gene || selectedTarget?.gene || selectedTargetId,
        smiles: prediction.compound.input_smiles, canonical_smiles: prediction.compound.canonical_smiles,
        pIC50: prediction.prediction.pIC50, IC50_nM: prediction.prediction.IC50_nM,
        uncertainty: prediction.uncertainty.prediction_sd,
        similarity: prediction.applicability_domain.max_tanimoto_similarity,
        confidence: prediction.confidence, model_version: prediction.model.version,
      }, ...previous].slice(0, 200));

    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to QSAR-X API. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };


  // =========================

  const predictBatch = async () => {
    if (!batchFile || !selectedTargetId) {
      setBatchError("Select a target and a CSV file first.");
      return;
    }
    setBatchLoading(true);
    setBatchError("");
    setBatchResult(null);
    try {
      const formData = new FormData();
      formData.append("file", batchFile);
      formData.append("target_id", selectedTargetId);
      const response = await fetch(`${API_URL}/predict-batch`, { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) {
        const detail = data.detail;
        throw new Error((typeof detail === "string" ? detail : detail?.message) || "Batch prediction failed.");
      }
      setBatchResult(data.result);
    } catch (batchPredictionError) {
      setBatchError(batchPredictionError.message || "Batch prediction could not be completed.");
    } finally {
      setBatchLoading(false);
    }
  };

  const downloadCsv = (rows, fileName) => {
    if (!Array.isArray(rows) || rows.length === 0) return;
    const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
    const value = (input) => {
      const text = input === null || input === undefined ? "" : String(input);
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => value(row[column])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = fileName; document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  };

  const downloadSingleResult = () => {
    if (!result) return;
    downloadCsv([{
      target_id: selectedTargetId, target_gene: result.target?.gene || selectedTarget?.gene,
      input_smiles: result.compound.input_smiles, canonical_smiles: result.compound.canonical_smiles,
      pIC50: result.prediction.pIC50, IC50_nM: result.prediction.IC50_nM,
      typical_error_pIC50: result.uncertainty.prediction_sd,
      max_tanimoto_similarity: result.applicability_domain.max_tanimoto_similarity,
      confidence: result.confidence, model_version: result.model.version,
    }], `${result.target?.gene || "qsar"}_prediction.csv`);
  };

  const downloadBatchResults = () => {
    const rows = batchResult?.results;
    if (!Array.isArray(rows) || rows.length === 0) return;
    const columns = ["row", "target_id", "smiles", "canonical_smiles", "pIC50", "IC50_nM", "uncertainty", "uncertainty_status", "uncertainty_method", "max_tanimoto_similarity", "applicability_domain", "confidence", "status", "error"];
    const csvValue = (value) => {
      const text = value === null || value === undefined ? "" : String(value);
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    };
    const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => csvValue(row[column])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${selectedTarget?.gene || "qsar"}_batch_predictions.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const useExample = () => {
    setSmiles(exampleSmiles);
    setError("");
    setResult(null);
  };

  const clearAll = () => {
    setSmiles("");
    setResult(null);
    setError("");
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><img src="/qsar-x-icon.svg" alt="QSAR-X" /></div><div><h1>QSAR-X</h1><p>Target-specific activity prediction</p></div></div>
        <div className="connection"><span className={`connection-dot ${apiConnected ? "" : "offline"}`}></span>{apiConnected ? "System online" : "System offline"}</div>
      </header>
      <main className="container">
        <div className="workspace-heading">
          <div><span className="eyebrow">SCREENING WORKSPACE</span><h2>Activity prediction</h2><p>Select a target, enter a compound, and run the target-specific QSAR model.</p></div>
          <div className="workspace-count"><strong>{targets.length}</strong><span>target models online</span></div>
        </div>
        <section className="workspace-grid">
          <article className="work-card target-work-card">
            <div className="card-title"><span className="step">01</span><div><h3>Target</h3><p>Choose the protein model.</p></div></div>
            <label htmlFor="target-picker">Target model</label>
            <select id="target-picker" className="target-select" value={selectedTargetId} onChange={(event) => { const target = targetOptions.find((item) => item.target_id === event.target.value); if (!target) return; setTargetQuery(target.gene); setSelectedTargetId(target.target_id); setResult(null); setError(""); }}>
              <option value="">Select a target</option>
              {targetOptions.map((target) => <option key={target.target_id} value={target.target_id}>{target.gene} · {target.name}</option>)}
            </select>
            {selectedCatalogTarget ? <div className="selected-target"><div className="target-symbol">{selectedCatalogTarget.gene}</div><div><strong>{selectedCatalogTarget.name}</strong><span>{selectedCatalogTarget.target_id} · {selectedCatalogTarget.endpoint}</span></div><span className="model-chip">{selectedTarget?.model_version || "Model ready"}</span></div> : <div className="selection-empty">Select one of the available target-specific models.</div>}
            <details className="advanced-target"><summary>Resolve by gene, UniProt ID, or protein sequence</summary><div className="resolve-row"><input value={targetQuery} placeholder="EGFR, P00533, CHEMBL203, or protein sequence" onChange={(event) => setTargetQuery(event.target.value)} /><button className="quiet-button" type="button" onClick={resolveProteinTarget} disabled={targetResolving || !apiConnected}>{targetResolving ? "Resolving…" : "Resolve"}</button></div>{targetMessage && <p className="target-message">{targetMessage}</p>}</details>
          </article>
          <article className="work-card compound-work-card">
            <div className="card-title"><span className="step">02</span><div><h3>Compound</h3><p>Paste a valid molecular SMILES.</p></div></div>
            <label htmlFor="smiles">SMILES</label>
            <textarea id="smiles" value={smiles} onChange={(event) => setSmiles(event.target.value)} placeholder="Example: CCOc1ccc2nc(S(N)(=O)=O)sc2c1" />
            <div className="compound-actions"><button className="text-button" type="button" onClick={useExample}>Load example</button><button className="text-button" type="button" onClick={clearAll}>Clear</button><button className="predict-button" type="button" onClick={predictActivity} disabled={loading || !canPredict}>{loading ? <><span className="spinner"></span>Running model</> : <>Run prediction <span>→</span></>}</button></div>
            {error && <div className="error-box">{error}</div>}
            <details className="batch-tool">
              <summary>Batch CSV prediction</summary>
              <p>Upload a CSV with a <code>smiles</code> column.</p>
              <label className="csv-upload" htmlFor="batch-file">{batchFile ? batchFile.name : "Choose CSV file"}<input id="batch-file" type="file" accept=".csv,text/csv" onChange={(event) => { setBatchFile(event.target.files?.[0] || null); setBatchResult(null); setBatchError(""); }} /></label>
              <div className="batch-run"><button className="quiet-button" type="button" onClick={predictBatch} disabled={batchLoading || !batchFile || !canPredict}>{batchLoading ? "Running batch…" : "Run batch"}</button>{batchResult && <><span>{batchResult.successful_predictions} / {batchResult.total_compounds} predictions completed</span><button className="download-button" type="button" onClick={downloadBatchResults}>Download results CSV</button></>}</div>
              {batchError && <div className="error-box">{batchError}</div>}
            </details>
          </article>
        </section>
        {result ? <section className="result-section">
          <div className="result-section-header"><div><span className="eyebrow">PREDICTION OUTPUT</span><h2>{result.target?.gene || selectedTarget?.gene || "Target"} activity</h2></div><div className="result-actions"><span className="model-chip">{result.confidence?.replaceAll("_", " ")}</span><button className="download-button" type="button" onClick={downloadSingleResult}>Download CSV</button></div></div>
          <div className="result-panel">
            <div className="structure-pane"><div className="molecule" dangerouslySetInnerHTML={{__html: result.compound.structure_svg}} /><code>{result.compound.canonical_smiles}</code></div>
            <div className="activity-pane"><span className="prediction-label">PREDICTED ACTIVITY</span><div className="pic50">{result.prediction.pIC50}</div><span className="pic50-label">pIC50</span><div className="ic50-box"><span>Predicted IC50</span><strong>{result.prediction.IC50_nM} nM</strong></div></div>
            <div className="evidence-pane"><Evidence label="Typical error" value={`${result.uncertainty.prediction_sd} pIC50`} /><Evidence label="Nearest similarity" value={result.applicability_domain.max_tanimoto_similarity} /><Evidence label="Model" value={result.model.version} /><details><summary>Prediction notes</summary><p>{result.warning || "No additional model note."}</p></details></div>
          </div>
        </section> : <section className="empty-result"><div className="empty-result-mark">⌁</div><div><strong>Prediction output will appear here</strong><span>Select a target and run a SMILES prediction to begin.</span></div></section>}
        <section className="history-strip">
          <div><strong>Previous predictions</strong><span>{history.length} saved in this browser</span></div>
          {history.length > 0 && <button className="quiet-button" type="button" onClick={() => downloadCsv(history, "qsar_prediction_history.csv")}>Download history CSV</button>}
          {history.length > 0 && <details><summary>View saved predictions</summary><div className="history-list">{history.map((item) => <div className="history-row" key={item.id}><span>{item.target_gene}</span><code>{item.smiles}</code><strong>{item.pIC50} pIC50</strong><small>{item.IC50_nM} nM · {item.createdAt}</small></div>)}</div></details>}
        </section>
      </main>
    </div>
  );
}


function Evidence({ label, value }) {
  return <div className="evidence-item"><span>{label}</span><strong>{value}</strong></div>;
}

export default App;



