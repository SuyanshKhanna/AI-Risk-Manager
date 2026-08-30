"""Decision Engine Service."""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import structlog

from omegaconf import DictConfig

from libs.common import (
    schemas, utils, metrics,
    generate_request_id, ServiceMetrics,
    track_latency, trace_operation,
)

logger = structlog.get_logger(__name__)


class Rule:
    """A single decision rule."""
    
    def __init__(
        self,
        name: str,
        condition: str,
        action: schemas.Action,
        priority: int,
        vector: schemas.FraudVector = None,
    ):
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.vector = vector
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition."""
        try:
            # Simple condition evaluation (in production use a proper rule engine)
            # Conditions like "upi_score > 0.7" or "voice_score > 0.9 and kyc_score > 0.8"
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"Rule evaluation failed: {self.name}, error: {e}")
            return False


class DecisionEngine:
    """Decision Engine that combines model scores with business rules."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.rules: List[Rule] = []
        self.metrics = ServiceMetrics("decision_engine")
        self._initialized = False
    
    async def initialize(self):
        """Initialize the decision engine."""
        if self._initialized:
            return
        
        logger.info("Initializing Decision Engine")
        
        # Load rules from config
        self._load_rules()
        
        self._initialized = True
        logger.info("Decision Engine initialized")
    
    def _load_rules(self):
        """Load rules from configuration."""
        rules_config = self.config.get("rules", [])
        
        for rule_config in rules_config:
            rule = Rule(
                name=rule_config["name"],
                condition=rule_config["condition"],
                action=schemas.Action(rule_config["action"]),
                priority=rule_config["priority"],
                vector=schemas.FraudVector(rule_config["vector"]) if rule_config.get("vector") else None,
            )
            self.rules.append(rule)
        
        # Sort by priority (higher first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        
        logger.info(f"Loaded {len(self.rules)} rules")
    
    @track_latency(metrics.GRPC_REQUEST_LATENCY, labels={"service": "decision_engine", "method": "decide"})
    async def decide(self, request: schemas.DecisionRequest) -> schemas.DecisionResponse:
        """Make a decision based on model scores and rules."""
        start_time = time.perf_counter()
        
        with trace_operation("decision_engine_decide", {
            "request_id": request.request_id,
            "merchant_id": request.merchant_id,
        }):
            # Prepare context for rule evaluation
            context = self._prepare_context(request)
            
            # Evaluate rules in priority order
            applied_rules = []
            final_action = schemas.Action.ALLOW
            final_score = 0.0
            vector_contributions = {}
            
            for rule in self.rules:
                if rule.evaluate(context):
                    applied_rules.append(rule.name)
                    final_action = rule.action
                    # For blocking rules, we can stop early
                    if rule.action in [schemas.Action.BLOCK, schemas.Action.SHADOW_BAN]:
                        break
            
            # Compute final score as weighted average of vector scores
            if request.vector_scores:
                total_weight = 0
                weighted_sum = 0
                for vector, score in request.vector_scores.items():
                    # Weight by vector importance (could be configured)
                    weight = self._get_vector_weight(vector)
                    weighted_sum += score * weight
                    total_weight += weight
                    vector_contributions[vector] = score
                
                final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Compute SHAP values (simplified)
            shap_values = self._compute_shap(request, context)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            response = schemas.DecisionResponse(
                request_id=request.request_id,
                final_action=final_action,
                final_score=final_score,
                applied_rules=applied_rules,
                vector_contributions=vector_contributions,
                shap_values=shap_values,
                model_version=self.config.get("model_version", "0.1.0"),
                latency_ms=latency_ms,
            )
            
            # Record metrics
            self.metrics.record_request("POST", "/decide", 200, latency_ms / 1000)
            self.metrics.record_grpc_request("Decide", "success", latency_ms / 1000)
            
            return response
    
    def _prepare_context(self, request: schemas.DecisionRequest) -> Dict[str, Any]:
        """Prepare context for rule evaluation."""
        context = {
            "merchant_id": request.merchant_id,
            **request.context,
        }
        
        # Add vector scores to context
        for vector, score in request.vector_scores.items():
            context[f"{vector.value}_score"] = score
        
        return context
    
    def _get_vector_weight(self, vector: schemas.FraudVector) -> float:
        """Get weight for a fraud vector."""
        weights = {
            schemas.FraudVector.UPI: 1.0,
            schemas.FraudVector.VOICE: 1.0,
            schemas.FraudVector.KYC: 1.0,
            schemas.FraudVector.CHARGEBACK: 0.8,
            schemas.FraudVector.RETURN: 0.8,
            schemas.FraudVector.REVIEW: 0.5,
        }
        return weights.get(vector, 0.5)
    
    def _compute_shap(
        self,
        request: schemas.DecisionRequest,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute SHAP values for explainability."""
        shap_values = {}
        
        # Add rule contributions
        for rule_name in request.applied_rules:
            shap_values[f"rule_{rule_name}"] = 1.0
        
        # Add vector contributions
        for vector, score in request.vector_scores.items():
            weight = self._get_vector_weight(vector)
            contribution = score * weight
            shap_values[f"vector_{vector.value}"] = contribution
        
        return shap_values


async def serve(config: DictConfig):
    """Main serving function."""
    engine = DecisionEngine(config)
    await engine.initialize()
    
    logger.info("Decision Engine serving started")
    
    # Keep running
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    import asyncio
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("configs/base.yaml")
    asyncio.run(serve(config))