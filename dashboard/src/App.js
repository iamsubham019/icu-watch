import React, { useState, useEffect } from 'react';
import './App.css';

const API_URL = 'https://icu-watch.onrender.com';

function SeverityBadge({ severity }) {
  const colors = {
    critical: { bg: '#ffe5e5', text: '#c0392b', border: '#e74c3c', icon: '🔴' },
    watch:    { bg: '#fff8e1', text: '#d68910', border: '#f39c12', icon: '🟡' },
    stable:   { bg: '#e8f8f5', text: '#1e8449', border: '#2ecc71', icon: '🟢' },
  };
  const style = colors[severity] || colors.stable;
  return (
    <span style={{
      backgroundColor: style.bg,
      color: style.text,
      border: `1px solid ${style.border}`,
      borderRadius: '20px',
      padding: '4px 12px',
      fontWeight: 'bold',
      fontSize: '13px',
    }}>
      {style.icon} {severity.toUpperCase()}
    </span>
  );
}

function VitalItem({ label, value, unit }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '8px 12px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      minWidth: '70px',
    }}>
      <span style={{ fontSize: '11px', color: '#7f8c8d', marginBottom: '2px' }}>{label}</span>
      <span style={{ fontSize: '16px', fontWeight: 'bold', color: '#2c3e50' }}>{value}</span>
      <span style={{ fontSize: '10px', color: '#95a5a6' }}>{unit}</span>
    </div>
  );
}
function PatientModal({ patientId, onClose }) {
  const [patient, setPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const [vitalsRes, predictRes] = await Promise.all([
        fetch(`${API_URL}/patient/${patientId}/vitals`),
        fetch(`${API_URL}/patient/${patientId}/predict`),
      ]);
      setPatient(await vitalsRes.json());
      setPrediction(await predictRes.json());
    };
    fetchData();
  }, [patientId]);

  if (!patient || !prediction) return (
    <div style={modalOverlayStyle}>
      <div style={modalStyle}>
        <p>Loading...</p>
      </div>
    </div>
  );

  const severity = prediction.severity;
  const borderColors = { critical: '#e74c3c', watch: '#f39c12', stable: '#2ecc71' };
  const borderColor = borderColors[severity];

  return (
    <div style={modalOverlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h2 style={{ margin: 0, color: '#2c3e50' }}>{patientId}</h2>
            <span style={{ fontSize: '13px', color: '#7f8c8d' }}>Age: {patient.age}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <SeverityBadge severity={severity} />
            <button onClick={onClose} style={{
              background: 'none', border: '1px solid #bdc3c7',
              borderRadius: '8px', padding: '6px 12px',
              cursor: 'pointer', fontSize: '13px', color: '#7f8c8d'
            }}>✕ Close</button>
          </div>
        </div>

        {/* Risk Bar */}
        <div style={{ marginBottom: '20px', padding: '16px', backgroundColor: '#f8f9fa', borderRadius: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontWeight: 'bold', color: '#2c3e50' }}>Deterioration Risk (6h)</span>
            <span style={{ fontWeight: 'bold', color: borderColor, fontSize: '18px' }}>
              {Math.round(prediction.deterioration_probability * 100)}%
            </span>
          </div>
          <div style={{ backgroundColor: '#ecf0f1', borderRadius: '6px', height: '12px' }}>
            <div style={{
              width: `${prediction.deterioration_probability * 100}%`,
              backgroundColor: borderColor,
              height: '12px', borderRadius: '6px',
              transition: 'width 0.5s ease',
            }} />
          </div>
          <div style={{ marginTop: '8px', fontSize: '12px', color: '#7f8c8d' }}>
            Confidence: {Math.round(prediction.confidence * 100)}%
          </div>
        </div>

        {/* Full Vitals */}
        <div style={{ marginBottom: '20px' }}>
          <h4 style={{ margin: '0 0 12px 0', color: '#2c3e50' }}>📊 Vital Signs</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
            <VitalItem label="Heart Rate"    value={patient.vitals?.heart_rate}       unit="bpm" />
            <VitalItem label="Systolic BP"   value={patient.vitals?.systolic_bp}      unit="mmHg" />
            <VitalItem label="Diastolic BP"  value={patient.vitals?.diastolic_bp}     unit="mmHg" />
            <VitalItem label="SpO2"          value={patient.vitals?.spo2}             unit="%" />
            <VitalItem label="Resp Rate"     value={patient.vitals?.respiratory_rate} unit="/min" />
            <VitalItem label="Temperature"   value={patient.vitals?.temperature}      unit="°C" />
            <VitalItem label="GCS"           value={patient.vitals?.gcs}              unit="/15" />
          </div>
        </div>

        {/* SHAP Chart */}
        <div>
          <h4 style={{ margin: '0 0 12px 0', color: '#2c3e50' }}>🔍 Risk Drivers (SHAP)</h4>
          {prediction.shap_values &&
            Object.entries(prediction.shap_values)
              .sort((a, b) => b[1] - a[1])
              .map(([vital, value], i) => (
                <div key={i} style={{ marginBottom: '10px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                    <span style={{ fontSize: '13px', color: '#2c3e50', fontWeight: '500' }}>{vital}</span>
                    <span style={{ fontSize: '13px', color: '#e74c3c', fontWeight: 'bold' }}>
                      +{value.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ backgroundColor: '#ecf0f1', borderRadius: '4px', height: '8px' }}>
                    <div style={{
                      width: `${Math.min(value * 300, 100)}%`,
                      backgroundColor: i === 0 ? '#e74c3c' : i === 1 ? '#e67e22' : '#3498db',
                      height: '8px', borderRadius: '4px',
                    }} />
                  </div>
                </div>
              ))
          }
        </div>

      </div>
    </div>
  );
}
const modalOverlayStyle = {
  position: 'fixed', top: 0, left: 0,
  width: '100%', height: '100%',
  backgroundColor: 'rgba(0,0,0,0.5)',
  display: 'flex', alignItems: 'center',
  justifyContent: 'center', zIndex: 1000,
};

const modalStyle = {
  backgroundColor: 'white',
  borderRadius: '16px',
  padding: '24px',
  width: '600px',
  maxHeight: '85vh',
  overflowY: 'auto',
  boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
};

function PatientCard({ patientId }) {
  const [patient, setPatient] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [vitalsRes, predictRes] = await Promise.all([
          fetch(`${API_URL}/patient/${patientId}/vitals`),
          fetch(`${API_URL}/patient/${patientId}/predict`),
        ]);
        const vitals = await vitalsRes.json();
        const predict = await predictRes.json();
        setPatient(vitals);
        setPrediction(predict);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [patientId]);

  if (loading) return (
    <div style={cardStyle('#f8f9fa')}>
      <p style={{ color: '#95a5a6' }}>Loading {patientId}...</p>
    </div>
  );

  const severity = prediction?.severity || 'stable';
  const borderColors = { critical: '#e74c3c', watch: '#f39c12', stable: '#2ecc71' };
  const borderColor = borderColors[severity];

  return (
    <>
      <div style={{...cardStyle(borderColor), cursor: 'pointer'}} onClick={() => setShowModal(true)}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div>
            <h3 style={{ margin: 0, color: '#2c3e50', fontSize: '16px' }}>{patientId}</h3>
            <span style={{ fontSize: '12px', color: '#7f8c8d' }}>Age: {patient?.age}</span>
          </div>
          <SeverityBadge severity={severity} />
        </div>

        {/* Risk probability bar */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ fontSize: '12px', color: '#7f8c8d' }}>Deterioration Risk</span>
            <span style={{ fontSize: '12px', fontWeight: 'bold', color: borderColor }}>
              {Math.round((prediction?.deterioration_probability || 0) * 100)}%
            </span>
          </div>
          <div style={{ backgroundColor: '#ecf0f1', borderRadius: '4px', height: '8px' }}>
            <div style={{
              width: `${(prediction?.deterioration_probability || 0) * 100}%`,
              backgroundColor: borderColor,
              height: '8px',
              borderRadius: '4px',
              transition: 'width 0.5s ease',
            }} />
          </div>
        </div>

        {/* Vitals */}
        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
          <VitalItem label="HR"    value={patient?.vitals?.heart_rate}        unit="bpm" />
          <VitalItem label="SBP"   value={patient?.vitals?.systolic_bp}       unit="mmHg" />
          <VitalItem label="SpO2"  value={patient?.vitals?.spo2}              unit="%" />
          <VitalItem label="Resp"  value={patient?.vitals?.respiratory_rate}  unit="/min" />
          <VitalItem label="Temp"  value={patient?.vitals?.temperature}       unit="°C" />
        </div>

        {/* SHAP Chart */}
        <div style={{ marginTop: '8px' }}>
          <span style={{ fontSize: '11px', color: '#7f8c8d', display: 'block', marginBottom: '6px' }}>
            Risk drivers:
          </span>
          {prediction?.shap_values &&
            Object.entries(prediction.shap_values)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 5)
              .map(([vital, value], i) => (
                <div key={i} style={{ marginBottom: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2px' }}>
                    <span style={{ fontSize: '10px', color: '#7f8c8d' }}>{vital}</span>
                    <span style={{ fontSize: '10px', color: '#e74c3c', fontWeight: 'bold' }}>
                      +{value.toFixed(3)}
                    </span>
                  </div>
                  <div style={{ backgroundColor: '#ecf0f1', borderRadius: '4px', height: '5px' }}>
                    <div style={{
                      width: `${Math.min(value * 300, 100)}%`,
                      backgroundColor: '#e74c3c',
                      height: '5px',
                      borderRadius: '4px',
                    }} />
                  </div>
                </div>
              ))
          }
        </div>

      </div>

      {showModal && (
        <PatientModal
          patientId={patientId}
          onClose={() => setShowModal(false)}
        />
      )}
    </>
  );
}
function statCardStyle(borderColor) {
  return {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '16px 24px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    borderTop: `4px solid ${borderColor}`,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    minWidth: '120px',
  };
}

function cardStyle(borderColor) {
  return {
    backgroundColor: 'white',
    borderRadius: '12px',
    padding: '16px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    borderLeft: `4px solid ${borderColor}`,
    transition: 'transform 0.2s',
  };
}



function App() {
  const [patients, setPatients] = useState([]);
  const [stats, setStats] = useState({ critical: 0, watch: 0, stable: 0 });
  const [severityMap, setSeverityMap] = useState({});
  const [loaded, setLoaded] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = async () => {
    setRefreshing(true);
    try {
      const res = await fetch(`${API_URL}/patients`);
      const data = await res.json();
      setPatients(data.patients);
      const preds = {};
      for (const pid of data.patients) {
        const r = await fetch(`${API_URL}/patient/${pid}/predict`);
        const pred = await r.json();
        preds[pid] = pred;
      }
      const critical = Object.values(preds).filter(p => p.severity === 'critical').length;
      const watch    = Object.values(preds).filter(p => p.severity === 'watch').length;
      const stable   = Object.values(preds).filter(p => p.severity === 'stable').length;
      setStats({ critical, watch, stable });
      setSeverityMap(Object.fromEntries(Object.entries(preds).map(([k, v]) => [k, v.severity])));
      setLoaded(true);
    } catch (err) {
      console.error(err);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f0f4f8', fontFamily: 'Inter, sans-serif' }}>
      {/* Header */}
      <div style={{
        backgroundColor: '#2c3e50',
        padding: '16px 32px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      }}>
        <span style={{ fontSize: '28px' }}>🏥</span>
        <div>
          <h1 style={{ margin: 0, color: 'white', fontSize: '22px' }}>ICU-Watch</h1>
          <p style={{ margin: 0, color: '#95a5a6', fontSize: '12px' }}>
            AI-Powered Patient Deterioration Monitoring
          </p>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '16px' }}>
          <span style={{ color: '#95a5a6', fontSize: '12px' }}>
            {patients.length} patients monitored • Updates every 30s
          </span>
          <button
            onClick={loadData}
            disabled={refreshing}
            style={{
              backgroundColor: refreshing ? '#95a5a6' : '#3498db',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              padding: '8px 16px',
              cursor: refreshing ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              fontWeight: 'bold',
            }}
          >
            {refreshing ? '⏳ Refreshing...' : '🔄 Refresh'}
          </button>
        </div>
      </div>

      {/* Stats Bar */}
      <div style={{
        padding: '16px 32px',
        display: 'flex',
        gap: '16px',
      }}>
        <div style={statCardStyle('#e74c3c')}>
          <span style={{ fontSize: '28px', fontWeight: 'bold', color: '#e74c3c' }}>{stats.critical}</span>
          <span style={{ fontSize: '13px', color: '#7f8c8d' }}>🔴 Critical</span>
        </div>
        <div style={statCardStyle('#f39c12')}>
          <span style={{ fontSize: '28px', fontWeight: 'bold', color: '#f39c12' }}>{stats.watch}</span>
          <span style={{ fontSize: '13px', color: '#7f8c8d' }}>🟡 Watch</span>
        </div>
        <div style={statCardStyle('#2ecc71')}>
          <span style={{ fontSize: '28px', fontWeight: 'bold', color: '#2ecc71' }}>{stats.stable}</span>
          <span style={{ fontSize: '13px', color: '#7f8c8d' }}>🟢 Stable</span>
        </div>
      </div>

      {/* Patient Grid */}
      <div style={{
        padding: '24px 32px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: '16px',
      }}>
        {patients
        .sort((a, b) => {
          const order = { critical: 0, watch: 1, stable: 2 };
          const aScore = order[severityMap[a] || 'stable'] ?? 2;
          const bScore = order[severityMap[b] || 'stable'] ?? 2;
          return aScore - bScore;
        })
        .map(pid => (
          <PatientCard key={pid} patientId={pid} />
        ))}
      </div>
    </div>
  );
}

export default App