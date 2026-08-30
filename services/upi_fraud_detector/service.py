"""UPI Fraud Detector Service."""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig

from libs.common import (
    schemas, utils, auth, metrics,
    generate_request_id, current_timestamp_ms, ServiceMetrics,
    track_latency, trace_operation,
)
from libs.feature_store import FeatureMaterializer, get_all_feature_views
from libs.str_estimator import STREstimator, STRConfig, STRClassifier

logger = structlog.get_logger(__name__)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for sequence modeling."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        sequence_length: int = 30,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, sequence_length, hidden_size) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Static feature projection
        self.static_projection = nn.Linear(input_dim, hidden_size)
        
        # Variable selection
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_dim),
            nn.Softmax(dim=-1),
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_dim),
        )
        
        # Gating for static context
        self.static_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        sequence: torch.Tensor,  # (batch, seq_len, input_dim)
        static: torch.Tensor,    # (batch, input_dim)
    ) -> torch.Tensor:
        batch_size = sequence.shape[0]
        
        # Project sequence
        seq_proj = self.input_projection(sequence) + self.pos_encoding
        
        # Transformer encoding
        seq_encoded = self.transformer(seq_proj)  # (batch, seq_len, hidden)
        
        # Use last timestep + mean pooling
        last_hidden = seq_encoded[:, -1, :]  # (batch, hidden)
        mean_hidden = seq_encoded.mean(dim=1)  # (batch, hidden)
        seq_context = torch.cat([last_hidden, mean_hidden], dim=-1)  # (batch, 2*hidden)
        
        # Static context
        static_proj = self.static_projection(static)  # (batch, hidden)
        static_gate = self.static_gate(static_proj)
        static_context = static_proj * static_gate
        
        # Combine
        combined = torch.cat([seq_context, static_context], dim=-1)
        
        # Output
        output = self.output_head(combined)
        return torch.sigmoid(output)


class UpiFraudDetector:
    """UPI Fraud Detection Service."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.str_estimator = None
        self.feature_materializer = None
        self.metrics = ServiceMetrics("upi_fraud_detector")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return
        
        logger.info("Initializing UPI Fraud Detector")
        
        # Initialize feature store
        self.feature_materializer = FeatureMaterializer()
        
        # Load model
        await self._load_model()
        
        # Initialize STR estimator
        self.str_estimator = STREstimator(STRConfig(
            propensity_model="lightgbm",
            outcome_model="lightgbm",
            corruption_model="beta_binomial",
            shrinkage_enabled=True,
        ))
        
        self._initialized = True
        logger.info("UPI Fraud Detector initialized")
    
    async def _load_model(self):
        """Load the trained model."""
        model_config = self.config.get("model", {})
        
        self.model = TemporalFusionTransformer(
            input_dim=model_config.get("input_dim", 100),
            hidden_size=model_config.get("hidden_size", 256),
            num_heads=model_config.get("num_heads", 8),
            num_layers=model_config.get("num_layers", 4),
            dropout=model_config.get("dropout", 0.1),
            sequence_length=model_config.get("sequence_length", 30),
        ).to(self.device)
        
        # Load weights if available
        model_path = model_config.get("model_path")
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")
        
        self.model.eval()
    
    @track_latency(metrics.MODEL_INFERENCE_LATENCY, labels={"model": "tft", "vector": "upi"})
    async def score_transaction(self, request: schemas.UpiTxnRequest) -> schemas.UpiTxnResponse:
        """Score a UPI transaction for fraud risk."""
        start_time = time.perf_counter()
        request_id = request.request_id or generate_request_id()
        
        with trace_operation("upi_fraud_score", {
            "merchant_id": request.merchant_id,
            "txn_id": request.txn_id,
            "amount": request.amount_paise,
        }):
            # Extract features
            features = await self._extract_features(request)
            
            # Get model prediction
            risk_score = await self._predict(features)
            
            # Determine action
            action, reasons = self._determine_action(risk_score, request)
            
            # Compute SHAP values (simplified)
            shap_values = await self._compute_shap(features)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            response = schemas.UpiTxnResponse(
                request_id=request_id,
                risk_score=float(risk_score),
                action=action,
                reasons=reasons,
                shap_values=shap_values,
                model_version=self.config.get("model_version", "0.1.0"),
                latency_ms=latency_ms,
                vector=schemas.FraudVector.UPI,
                settlement_verified=features.get("settlement_verified", False),
                qr_tamper_detected=features.get("qr_tamper_detected", False),
                velocity_anomaly=features.get("velocity_anomaly", False),
            )
            
            # Record metrics
            self.metrics.record_score("upi", risk_score, action.value)
            self.metrics.record_grpc_request("ScoreUpiTxn", "success", latency_ms / 1000)
            
            return response
    
    async def _extract_features(self, request: schemas.UpiTxnRequest) -> Dict[str, Any]:
        """Extract features for a transaction."""
        features = {}
        
        # Settlement verification
        settlement_verified = False
        if request.pg_callback_status and request.pg_callback_ts_ms and request.screenshot_ts_ms:
            delta_ms = request.pg_callback_ts_ms - request.screenshot_ts_ms
            settlement_verified = (
                request.pg_callback_status == "SUCCESS" and
                abs(delta_ms) < 60000  # Within 1 minute
            )
        features["settlement_verified"] = settlement_verified
        features["settlement_delta_ms"] = delta_ms if settlement_verified else None
        
        # QR tamper detection (placeholder - would use YOLOv8-seg in production)
        qr_tamper_detected = False
        if request.qr_image_b64:
            # In production: perceptual hash comparison + overlay detection
            qr_tamper_detected = await self._detect_qr_tamper(request.qr_image_b64)
        features["qr_tamper_detected"] = qr_tamper_detected
        
        # Velocity features from feature store
        velocity_features = await self._get_velocity_features(request)
        features.update(velocity_features)
        features["velocity_anomaly"] = velocity_features.get("is_velocity_anomaly", False)
        
        # Device features
        device_features = await self._get_device_features(request.device_fingerprint)
        features.update(device_features)
        
        # Network features
        network_features = await self._get_network_features(request)
        features.update(network_features)
        
        # Merchant features
        merchant_features = await self._get_merchant_features(request.merchant_id)
        features.update(merchant_features)
        
        # Customer features
        customer_features = await self._get_customer_features(request.upi_handle)
        features.update(customer_features)
        
        # Sequence features (30-day history)
        sequence_features = await self._get_sequence_features(request)
        features["sequence"] = sequence_features
        
        return features
    
    async def _detect_qr_tamper(self, qr_image_b64: str) -> bool:
        """Detect QR code tampering (placeholder)."""
        # In production:
        # 1. Decode base64 image
        # 2. Compute perceptual hash (pHash)
        # 3. Compare with registered QR pHash for merchant
        # 4. Run YOLOv8-seg to detect overlays
        # 5. Return True if tampering detected
        return False
    
    async def _get_velocity_features(self, request: schemas.UpiTxnRequest) -> Dict[str, Any]:
        """Get velocity features from feature store."""
        try:
            entity_rows = [
                {"upi_handle": request.upi_handle},
                {"device_fingerprint": request.device_fingerprint},
            ]
            features = self.feature_materializer.get_online_features(
                entity_rows=entity_rows,
                features=[
                    "upi_handle:upi_txn_count_5min",
                    "upi_handle:upi_txn_count_1hr",
                    "upi_handle:upi_refund_count_1hr",
                    "upi_handle:upi_amount_zscore_1hr",
                    "upi_handle:upi_unique_merchants_1hr",
                    "upi_handle:upi_unique_devices_1hr",
                    "device:device_merchant_count_30d",
                    "device:device_customer_count_30d",
                ],
            )
            
            # Process features
            is_anomaly = False
            if features.get("upi_handle:upi_txn_count_5min", [0])[0] > 10:
                is_anomaly = True
            if features.get("upi_handle:upi_amount_zscore_1hr", [0])[0] > 3:
                is_anomaly = True
            
            return {
                "velocity_features": features,
                "is_velocity_anomaly": is_anomaly,
            }
        except Exception as e:
            logger.warning(f"Failed to get velocity features: {e}")
            return {"is_velocity_anomaly": False}
    
    async def _get_device_features(self, device_fingerprint: str) -> Dict[str, Any]:
        """Get device features from feature store."""
        try:
            entity_rows = [{"device_fingerprint": device_fingerprint}]
            features = self.feature_materializer.get_online_features(
                entity_rows=entity_rows,
                features=[
                    "device:device_os",
                    "device:device_model",
                    "device:device_sensor_entropy",
                    "device:device_app_tamper_score",
                    "device:device_emulator_score",
                    "device:device_root_score",
                    "device:device_vpn_probability",
                    "device:device_fraud_reports_30d",
                ],
            )
            return {"device_features": features}
        except Exception as e:
            logger.warning(f"Failed to get device features: {e}")
            return {}
    
    async def _get_network_features(self, request: schemas.UpiTxnRequest) -> Dict[str, Any]:
        """Get network features (IP reputation, etc.)."""
        # Placeholder - would integrate with IPQualityScore or similar
        return {
            "ip_reputation_score": 0.0,
            "is_vpn": False,
            "is_proxy": False,
            "asn_reputation": 0.0,
        }
    
    async def _get_merchant_features(self, merchant_id: str) -> Dict[str, Any]:
        """Get merchant features from feature store."""
        try:
            entity_rows = [{"merchant_id": merchant_id}]
            features = self.feature_materializer.get_online_features(
                entity_rows=entity_rows,
                features=[
                    "merchant:merchant_category",
                    "merchant:merchant_gmv_30d",
                    "merchant:merchant_txn_count_30d",
                    "merchant:merchant_chargeback_rate_30d",
                    "merchant:merchant_refund_rate_30d",
                    "merchant:merchant_avg_order_value",
                    "merchant:merchant_churn_risk_score",
                    "merchant:merchant_risk_tier",
                    "merchant:merchant_fraud_rate_30d",
                ],
            )
            return {"merchant_features": features}
        except Exception as e:
            logger.warning(f"Failed to get merchant features: {e}")
            return {}
    
    async def _get_customer_features(self, upi_handle: str) -> Dict[str, Any]:
        """Get customer features from feature store."""
        try:
            # Would need customer_id, using upi_handle as proxy
            return {}
        except Exception as e:
            logger.warning(f"Failed to get customer features: {e}")
            return {}
    
    async def _get_sequence_features(self, request: schemas.UpiTxnRequest) -> np.ndarray:
        """Get 30-day sequence features for TFT."""
        # Placeholder - would fetch last 30 days of transaction history
        # and create sequence tensor of shape (30, input_dim)
        seq_len = self.config.model.get("sequence_length", 30)
        input_dim = self.config.model.get("input_dim", 100)
        return np.zeros((seq_len, input_dim), dtype=np.float32)
    
    async def _predict(self, features: Dict[str, Any]) -> float:
        """Run model inference."""
        # Prepare inputs
        sequence = features.get("sequence", np.zeros((30, 100)))
        static = self._create_static_vector(features)
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        static_tensor = torch.FloatTensor(static).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(sequence_tensor, static_tensor)
            risk_score = output.item()
        
        return risk_score
    
    def _create_static_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Create static feature vector."""
        # Combine all non-sequence features into a single vector
        vec = []
        
        # Settlement
        vec.append(1.0 if features.get("settlement_verified") else 0.0)
        vec.append(features.get("settlement_delta_ms", 0) / 100000.0)  # Normalized
        
        # QR
        vec.append(1.0 if features.get("qr_tamper_detected") else 0.0)
        
        # Velocity
        vel = features.get("velocity_features", {})
        vec.append(vel.get("upi_handle:upi_txn_count_5min", [0])[0] / 20.0)
        vec.append(vel.get("upi_handle:upi_txn_count_1hr", [0])[0] / 100.0)
        vec.append(vel.get("upi_handle:upi_refund_count_1hr", [0])[0] / 10.0)
        vec.append(np.clip(vel.get("upi_handle:upi_amount_zscore_1hr", [0])[0] / 5.0, -1, 1))
        vec.append(vel.get("upi_handle:upi_unique_merchants_1hr", [0])[0] / 10.0)
        vec.append(vel.get("upi_handle:upi_unique_devices_1hr", [0])[0] / 5.0)
        
        # Device
        dev = features.get("device_features", {})
        vec.append(dev.get("device:device_sensor_entropy", [0.5])[0])
        vec.append(dev.get("device:device_app_tamper_score", [0])[0])
        vec.append(dev.get("device:device_emulator_score", [0])[0])
        vec.append(dev.get("device:device_root_score", [0])[0])
        vec.append(dev.get("device:device_vpn_probability", [0])[0])
        vec.append(dev.get("device:device_fraud_reports_30d", [0])[0] / 10.0)
        
        # Network
        vec.append(features.get("ip_reputation_score", 0))
        vec.append(1.0 if features.get("is_vpn") else 0.0)
        vec.append(1.0 if features.get("is_proxy") else 0.0)
        vec.append(features.get("asn_reputation", 0))
        
        # Merchant
        mer = features.get("merchant_features", {})
        vec.append(mer.get("merchant:merchant_gmv_30d", [0])[0] / 1e8)
        vec.append(mer.get("merchant:merchant_txn_count_30d", [0])[0] / 10000.0)
        vec.append(mer.get("merchant:merchant_chargeback_rate_30d", [0])[0])
        vec.append(mer.get("merchant:merchant_refund_rate_30d", [0])[0])
        vec.append(mer.get("merchant:merchant_avg_order_value", [0])[0] / 100000.0)
        vec.append(mer.get("merchant:merchant_churn_risk_score", [0])[0])
        vec.append(mer.get("merchant:merchant_fraud_rate_30d", [0])[0])
        
        # Pad or truncate to input_dim
        input_dim = self.config.model.get("input_dim", 100)
        vec = vec[:input_dim] + [0.0] * max(0, input_dim - len(vec))
        
        return np.array(vec, dtype=np.float32)
    
    def _determine_action(
        self,
        risk_score: float,
        request: schemas.UpiTxnRequest,
    ) -> tuple[schemas.Action, List[str]]:
        """Determine action based on risk score."""
        threshold_challenge = self.config.get("serving", {}).get("threshold_challenge", 0.3)
        threshold_block = self.config.get("serving", {}).get("threshold_block", 0.7)
        
        reasons = []
        
        if risk_score >= threshold_block:
            action = schemas.Action.BLOCK
            reasons.append(f"High fraud risk score: {risk_score:.2f}")
            if not request.pg_callback_status == "SUCCESS":
                reasons.append("Settlement not verified")
            if request.qr_image_b64:  # Would check actual detection
                reasons.append("QR code tampering suspected")
        elif risk_score >= threshold_challenge:
            action = schemas.Action.CHALLENGE
            reasons.append(f"Elevated fraud risk score: {risk_score:.2f}")
            reasons.append("Additional verification required")
        else:
            action = schemas.Action.ALLOW
            reasons.append("Low fraud risk")
        
        return action, reasons
    
    async def _compute_shap(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Compute SHAP values for explainability (simplified)."""
        # In production: use SHAP library with model
        # For now, return feature importance based on heuristics
        shap_values = {}
        
        if not features.get("settlement_verified", False):
            shap_values["settlement_not_verified"] = 0.3
        
        if features.get("qr_tamper_detected", False):
            shap_values["qr_tamper"] = 0.4
        
        if features.get("velocity_anomaly", False):
            shap_values["velocity_anomaly"] = 0.25
        
        device_fraud = features.get("device_features", {}).get("device:device_fraud_reports_30d", [0])[0]
        if device_fraud > 0:
            shap_values["device_fraud_history"] = 0.2
        
        merchant_fraud = features.get("merchant_features", {}).get("merchant:merchant_fraud_rate_30d", [0])[0]
        if merchant_fraud > 0.01:
            shap_values["merchant_high_fraud_rate"] = 0.15
        
        return shap_values
    
    async def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train the model with STR label correction."""
        logger.info("Starting UPI fraud detector training")
        
        # Prepare features and labels
        X_train = self._prepare_features(train_data)
        y_train = train_data["label"].values
        R_train = train_data.get("authorized", pd.Series(1, index=train_data.index)).values
        O_train = train_data.get("reported", pd.Series(1, index=train_data.index)).values
        D_train = train_data.get("mature", pd.Series(1, index=train_data.index)).values
        groups_train = train_data.get("issuer_id", pd.Series(0, index=train_data.index)).values
        
        # Fit STR estimator for label correction
        logger.info("Fitting STR estimator for label correction")
        self.str_estimator.fit(
            X_train, y_train, R_train, O_train, D_train, groups_train
        )
        
        # Get corrected pseudo-labels
        pseudo_labels = self.str_estimator.pseudo_labels
        weights = self.str_estimator.get_training_weights()
        
        # Train TFT model on pseudo-labels
        logger.info("Training TFT model on STR-corrected labels")
        await self._train_tft(X_train, pseudo_labels, weights, val_data)
        
        logger.info("Training completed")
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from dataframe."""
        # Implementation would convert dataframe to feature matrix
        # matching the model's input_dim
        n_samples = len(data)
        input_dim = self.config.model.get("input_dim", 100)
        return np.random.randn(n_samples, input_dim).astype(np.float32)  # Placeholder
    
    async def _train_tft(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        weights: np.ndarray,
        val_data: pd.DataFrame,
    ):
        """Train the TFT model."""
        # Convert to sequences
        seq_len = self.config.model.get("sequence_length", 30)
        n_train = len(X_train)
        
        # Create sequence data (placeholder)
        X_seq = np.random.randn(n_train, seq_len, X_train.shape[1]).astype(np.float32)
        X_static = X_train.astype(np.float32)
        
        # Convert to tensors
        X_seq_tensor = torch.FloatTensor(X_seq).to(self.device)
        X_static_tensor = torch.FloatTensor(X_static).to(self.device)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        weight_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.get("learning_rate", 1e-4),
            weight_decay=0.01,
        )
        criterion = nn.BCELoss(reduction="none")
        
        # Training loop
        epochs = self.config.model.get("epochs", 50)
        batch_size = self.config.model.get("batch_size", 64)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for i in range(0, n_train, batch_size):
                batch_seq = X_seq_tensor[i:i+batch_size]
                batch_static = X_static_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                batch_w = weight_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_seq, batch_static)
                loss = (criterion(outputs, batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
        
        # Save model
        model_path = self.config.model.get("model_path", "models/upi_fraud_tft.pt")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")


async def serve(config: DictConfig):
    """Main serving function."""
    detector = UpiFraudDetector(config)
    await detector.initialize()
    
    # In production: start gRPC server
    logger.info("UPI Fraud Detector serving started")
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    import asyncio
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("configs/upi_local.yaml")
    asyncio.run(serve(config))