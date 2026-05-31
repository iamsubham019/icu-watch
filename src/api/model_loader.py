"""
Model Loader — connects Subham's trained LSTM to Swarnali's FastAPI
Authors: Swarnali Ghosh & Subham Pal
"""

import numpy as np
import joblib
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
PROCESSED_DIR = Path('data/processed')
MODEL_PATH    = PROCESSED_DIR / 'models' / 'lstm_best.pt'
SCALER_PATH   = PROCESSED_DIR / 'scaler.pkl'
FEATURES_PATH = PROCESSED_DIR / 'feature_cols.pkl'
CONFIG_PATH   = PROCESSED_DIR / 'config.json'

# ── Optional PyTorch import ───────────────────────────────────
try:
    import torch
    import torch.nn as nn

    class TemporalAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        def forward(self, lstm_out):
            scores  = self.attention(lstm_out)
            weights = torch.softmax(scores, dim=1)
            context = (weights * lstm_out).sum(dim=1)
            return context, weights.squeeze(-1)

    class ICUDeteriorationLSTM(nn.Module):
        def __init__(self, input_size=104, hidden_size=128, num_layers=2, dropout=0.3):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
                batch_first=True, bidirectional=True
            )
            self.attention  = TemporalAttention(hidden_size)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1)
            )
        def forward(self, x):
            x = self.input_norm(x)
            lstm_out, _ = self.lstm(x)
            context, attn = self.attention(lstm_out)
            return self.classifier(context).squeeze(-1), attn

    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False


class ModelLoader:
    def __init__(self):
        self.model        = None
        self.scaler       = None
        self.feature_cols = None
        self.config       = None
        self.device       = None
        self.loaded       = False
        self._load()

    def _load(self):
        if not TORCH_AVAILABLE:
            print('⚠️  PyTorch not available — using mock predictions')
            self.loaded = False
            return
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            with open(CONFIG_PATH) as f:
                self.config = json.load(f)

            self.scaler       = joblib.load(SCALER_PATH)
            self.feature_cols = joblib.load(FEATURES_PATH)

            n_features  = self.config['n_features']
            self.model  = ICUDeteriorationLSTM(input_size=n_features).to(self.device)
            checkpoint  = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()

            self.loaded = True
            print(f'✅ LSTM model loaded on {self.device}')
            print(f'   Val AUROC: {checkpoint["val_auroc"]:.4f}')

        except Exception as e:
            print(f'⚠️  Model not loaded: {e}')
            print('   Using mock predictions instead')
            self.loaded = False

    def predict(self, vitals: dict) -> dict:
        if not self.loaded:
            return self._mock_predict(vitals)
        try:
            import torch
            feature_vector = self._build_features(vitals)
            scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            window_size = self.config['window_size']
            sequence = np.repeat(scaled, window_size, axis=0).reshape(1, window_size, -1)
            X = torch.FloatTensor(sequence).to(self.device)
            with torch.no_grad():
                self.model.eval()
                logit, _ = self.model(X)
                prob = torch.sigmoid(logit).item()
            clinical_risk = self._mock_predict(vitals)['probability']
            blended_prob  = round((prob * 0.4) + (clinical_risk * 0.6), 4)
            return {'probability': blended_prob, 'model': 'lstm', 'top_hour': 0}
        except Exception as e:
            print(f'Prediction error: {e}')
            return self._mock_predict(vitals)

    def _build_features(self, vitals: dict) -> np.ndarray:
        base = {
            'HR':    vitals.get('heart_rate', 80),
            'O2Sat': vitals.get('spo2', 97),
            'Temp':  vitals.get('temperature', 37),
            'SBP':   vitals.get('systolic_bp', 120),
            'MAP':   (vitals.get('systolic_bp', 120) + 2 * vitals.get('diastolic_bp', 80)) / 3,
            'DBP':   vitals.get('diastolic_bp', 80),
            'Resp':  vitals.get('respiratory_rate', 16),
        }
        feature_vector = np.zeros(len(self.feature_cols))
        for i, col in enumerate(self.feature_cols):
            for vital_key, vital_val in base.items():
                if col.startswith(vital_key):
                    feature_vector[i] = vital_val
                    break
        return feature_vector

    def _mock_predict(self, vitals: dict) -> dict:
        import random
        risk = 0
        if vitals.get('heart_rate', 80) > 100:        risk += 0.2
        if vitals.get('systolic_bp', 120) < 100:      risk += 0.25
        if vitals.get('spo2', 97) < 95:               risk += 0.2
        if vitals.get('respiratory_rate', 16) > 20:   risk += 0.2
        if vitals.get('temperature', 37) > 38.5:      risk += 0.15
        risk = max(0.0, min(round(risk + random.uniform(-0.05, 0.05), 2), 1.0))
        return {'probability': risk, 'model': 'mock', 'top_hour': 0}


model_loader = ModelLoader()