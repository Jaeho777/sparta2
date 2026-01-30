"""
Advanced Deep Learning Models for Time Series Forecasting
==========================================================
고급 딥러닝 모델 구현:
1. Attention-based LSTM
2. Transformer Encoder
3. N-BEATS 스타일 모델
4. Temporal Convolutional Network (TCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# 1. Dataset 클래스
# =============================================================================

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋"""
    def __init__(self, X, y, seq_len=24):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


# =============================================================================
# 2. Attention LSTM
# =============================================================================

class AttentionLayer(nn.Module):
    """Self-Attention 레이어"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1), dim=1
        )  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_size)

        return context, attention_weights


class AttentionLSTM(nn.Module):
    """Attention-based LSTM"""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = AttentionLayer(hidden_size * 2)  # bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        context, _ = self.attention(lstm_out)  # (batch, hidden_size*2)
        out = F.relu(self.fc1(self.dropout(context)))
        out = self.fc2(out)
        return out.squeeze(-1)


# =============================================================================
# 3. Transformer Encoder
# =============================================================================

class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    """Transformer Encoder for forecasting"""
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x.squeeze(-1)


# =============================================================================
# 4. TCN (Temporal Convolutional Network)
# =============================================================================

class CausalConv1d(nn.Module):
    """인과적 1D 컨볼루션"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    """TCN 블록"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return F.relu(x + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, n_features, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.network(x)  # (batch, num_channels[-1], seq_len)
        x = x[:, :, -1]  # 마지막 시점만 사용
        x = self.fc(x)
        return x.squeeze(-1)


# =============================================================================
# 5. N-BEATS 스타일 모델
# =============================================================================

class NBeatsBlock(nn.Module):
    """N-BEATS 블록"""
    def __init__(self, input_size, theta_size, hidden_size=256, num_layers=4):
        super().__init__()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc = nn.Sequential(*layers)

        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc(x)
        backcast = self.theta_b(x)
        forecast = self.theta_f(x)
        return backcast, forecast


class NBeatsForecaster(nn.Module):
    """N-BEATS 스타일 예측 모델"""
    def __init__(self, input_size, hidden_size=128, num_blocks=3, num_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, input_size, hidden_size, num_layers)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> flatten
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # (batch, seq_len * n_features)

        forecast = torch.zeros(batch_size, 1, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(x)
            x = x - backcast
            forecast = forecast + block_forecast

        return forecast.squeeze(-1)


# =============================================================================
# 6. 학습 함수
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """모델 학습"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 최적 모델 로드
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def predict(model, data_loader):
    """예측"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.extend(y_pred.cpu().numpy())
    return np.array(predictions)


# =============================================================================
# 7. 딥러닝 파이프라인
# =============================================================================

def run_dl_pipeline(X_train, y_train, X_val, y_val, X_test, y_test,
                    seq_len=24, batch_size=32, epochs=100, verbose=True):
    """딥러닝 모델 파이프라인"""

    if verbose:
        print("\n" + "=" * 60)
        print("Deep Learning Models")
        print("=" * 60)

    # 스케일링
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    n_features = X_train_scaled.shape[1]

    # Dataset 생성
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train, seq_len)
    val_dataset = TimeSeriesDataset(
        np.vstack([X_train_scaled[-seq_len:], X_val_scaled]),
        pd.concat([y_train.iloc[-seq_len:], y_val]) if hasattr(y_train, 'iloc') else
        np.concatenate([y_train[-seq_len:], y_val]),
        seq_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 정의
    models = {
        'AttentionLSTM': AttentionLSTM(n_features, hidden_size=64, num_layers=2, dropout=0.2),
        'Transformer': TransformerForecaster(n_features, d_model=64, nhead=4, num_layers=2),
        'TCN': TCN(n_features, num_channels=[32, 64, 64], kernel_size=3),
    }

    # N-BEATS는 입력 크기가 다름
    input_size_nbeats = seq_len * n_features
    if input_size_nbeats < 10000:  # 메모리 제한
        models['NBeats'] = NBeatsForecaster(input_size_nbeats, hidden_size=128, num_blocks=3)

    results = []
    trained_models = {}

    for name, model in models.items():
        if verbose:
            print(f"\n  Training {name}...")

        try:
            trained_model, val_loss = train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=0.001, patience=15
            )

            # Validation 예측
            val_preds = predict(trained_model, val_loader)

            # 길이 맞추기
            actual_len = min(len(val_preds), len(y_val))
            y_val_arr = y_val.values[-actual_len:] if hasattr(y_val, 'values') else y_val[-actual_len:]
            val_preds = val_preds[-actual_len:]

            rmse = np.sqrt(mean_squared_error(y_val_arr, val_preds))
            mae = mean_absolute_error(y_val_arr, val_preds)

            results.append({
                'Model': f'DL_{name}',
                'RMSE': rmse,
                'MAE': mae,
                'Val_Loss': val_loss
            })

            trained_models[name] = trained_model

            if verbose:
                print(f"    Val RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")

    # Test 예측 (Train+Val로 재학습)
    if verbose:
        print("\n  Retraining on Train+Val and testing...")

    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = pd.concat([y_train, y_val]) if hasattr(y_train, 'iloc') else np.concatenate([y_train, y_val])

    train_val_dataset = TimeSeriesDataset(X_train_val, y_train_val, seq_len)
    test_dataset = TimeSeriesDataset(
        np.vstack([X_train_val[-seq_len:], X_test_scaled]),
        pd.concat([y_train_val.iloc[-seq_len:] if hasattr(y_train_val, 'iloc') else pd.Series(y_train_val[-seq_len:]), y_test])
        if hasattr(y_test, 'iloc') else np.concatenate([y_train_val[-seq_len:], y_test]),
        seq_len
    )

    train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_results = []
    test_predictions = {}

    for name in trained_models.keys():
        try:
            # 모델 재초기화 및 학습
            if name == 'AttentionLSTM':
                model = AttentionLSTM(n_features, hidden_size=64, num_layers=2, dropout=0.2)
            elif name == 'Transformer':
                model = TransformerForecaster(n_features, d_model=64, nhead=4, num_layers=2)
            elif name == 'TCN':
                model = TCN(n_features, num_channels=[32, 64, 64], kernel_size=3)
            elif name == 'NBeats':
                model = NBeatsForecaster(seq_len * n_features, hidden_size=128, num_blocks=3)

            trained_model, _ = train_model(
                model, train_val_loader, test_loader,
                epochs=epochs, lr=0.001, patience=15
            )

            test_preds = predict(trained_model, test_loader)

            # 길이 맞추기
            actual_len = min(len(test_preds), len(y_test))
            y_test_arr = y_test.values[-actual_len:] if hasattr(y_test, 'values') else y_test[-actual_len:]
            test_preds = test_preds[-actual_len:]

            rmse = np.sqrt(mean_squared_error(y_test_arr, test_preds))
            mae = mean_absolute_error(y_test_arr, test_preds)
            mape = np.mean(np.abs((y_test_arr - test_preds) / y_test_arr)) * 100

            test_results.append({
                'Model': f'DL_{name}',
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })

            test_predictions[f'DL_{name}'] = test_preds

            if verbose:
                print(f"    {name} Test RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        except Exception as e:
            if verbose:
                print(f"    {name} Test Error: {e}")

    return {
        'val_results': pd.DataFrame(results),
        'test_results': pd.DataFrame(test_results),
        'test_predictions': test_predictions
    }


if __name__ == '__main__':
    print("Deep Learning models module loaded.")
    print(f"Device: {device}")
