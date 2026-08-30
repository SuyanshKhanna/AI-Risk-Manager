.PHONY: help dev-up dev-down test lint typecheck format build docker-build docker-push train serve-local clean

# Default target
help:
	@echo "AI Risk Manager - Multi-Vector Fraud Detection Platform"
	@echo ""
	@echo "Available targets:"
	@echo "  dev-up        - Start local development stack (Kafka, Redis, Postgres, MinIO, Feast, MLflow)"
	@echo "  dev-down      - Stop local development stack"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-adversarial - Run adversarial robustness tests"
	@echo "  lint          - Run ruff linter"
	@echo "  typecheck     - Run mypy type checker"
	@echo "  format        - Format code with ruff"
	@echo "  build         - Build all Docker images"
	@echo "  docker-build  - Build Docker images for all services"
	@echo "  docker-push   - Push Docker images to registry"
	@echo "  train-upi     - Train UPI fraud detector (local)"
	@echo "  train-voice   - Train voice anti-spoofing model (local)"
	@echo "  train-kyc     - Train KYC liveness model (local)"
	@echo "  train-chargeback - Train chargeback responder (local)"
	@echo "  train-return  - Train return risk scorer (local)"
	@echo "  train-review  - Train review ring detector (local)"
	@echo "  serve-local   - Start all services locally (gRPC on :50051, HTTP on :8080)"
	@echo "  clean         - Clean build artifacts and caches"

# Local development stack
dev-up:
	docker-compose -f infra/docker-compose.yaml up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "Services started:"
	@docker-compose -f infra/docker-compose.yaml ps

dev-down:
	docker-compose -f infra/docker-compose.yaml down -v

# Testing
test: test-unit test-integration

test-unit:
	uv run pytest tests/unit -v -x --tb=short

test-integration:
	uv run pytest tests/integration -v -x --tb=short

test-adversarial:
	uv run pytest tests/adversarial -v -x --tb=short

test-cov:
	uv run pytest tests/unit tests/integration --cov=libs --cov=services --cov=pipelines --cov-report=term-missing --cov-report=html

# Code quality
lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

typecheck:
	uv run mypy .

# Security
security-scan:
	uv run bandit -r libs services pipelines -f json -o bandit-report.json || true
	uv run safety check --json --output safety-report.json || true

# Build
build: docker-build

docker-build:
	docker build -f docker/base.Dockerfile -t ai-risk-manager/base:latest .
	docker build -f docker/upi_fraud_detector.Dockerfile -t ai-risk-manager/upi-fraud-detector:latest .
	docker build -f docker/voice_auth.Dockerfile -t ai-risk-manager/voice-auth:latest .
	docker build -f docker/kyc_liveness.Dockerfile -t ai-risk-manager/kyc-liveness:latest .
	docker build -f docker/chargeback_responder.Dockerfile -t ai-risk-manager/chargeback-responder:latest .
	docker build -f docker/return_risk_scorer.Dockerfile -t ai-risk-manager/return-risk-scorer:latest .
	docker build -f docker/review_ring_detector.Dockerfile -t ai-risk-manager/review-ring-detector:latest .
	docker build -f docker/decision_engine.Dockerfile -t ai-risk-manager/decision-engine:latest .

docker-push:
	docker push ai-risk-manager/base:latest
	docker push ai-risk-manager/upi-fraud-detector:latest
	docker push ai-risk-manager/voice-auth:latest
	docker push ai-risk-manager/kyc-liveness:latest
	docker push ai-risk-manager/chargeback-responder:latest
	docker push ai-risk-manager/return-risk-scorer:latest
	docker push ai-risk-manager/review-ring-detector:latest
	docker push ai-risk-manager/decision-engine:latest

# Training (local)
train-upi:
	uv run python -m upi_fraud_detector.train --config configs/upi_local.yaml

train-voice:
	uv run python -m voice_auth.train --config configs/voice_local.yaml

train-kyc:
	uv run python -m kyc_liveness.train --config configs/kyc_local.yaml

train-chargeback:
	uv run python -m chargeback_responder.train --config configs/chargeback_local.yaml

train-return:
	uv run python -m return_risk_scorer.train --config configs/return_local.yaml

train-review:
	uv run python -m review_ring_detector.train --config configs/review_local.yaml

# Serving
serve-local:
	uv run python -m upi_fraud_detector.serve --config configs/upi_local.yaml &
	uv run python -m voice_auth.serve --config configs/voice_local.yaml &
	uv run python -m kyc_liveness.serve --config configs/kyc_local.yaml &
	uv run python -m chargeback_responder.serve --config configs/chargeback_local.yaml &
	uv run python -m return_risk_scorer.serve --config configs/return_local.yaml &
	uv run python -m review_ring_detector.serve --config configs/review_local.yaml &
	uv run python -m decision_engine.serve --config configs/decision_local.yaml &
	@echo "All services started. gRPC on :50051, HTTP on :8080"
	@wait

# Data generation
generate-data:
	uv run python scripts/data_gen/generate_synthetic_data.py --config configs/data_gen.yaml

# Clean
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
	rm -rf dist build *.egg-info
	rm -rf .coverage htmlcov bandit-report.json safety-report.json
	docker system prune -f

# Install
install:
	uv sync --all-extras

install-dev:
	uv sync --all-extras --dev

# Pre-commit
pre-commit-install:
	uv run pre-commit install

pre-commit-run:
	uv run pre-commit run --all-files

# CI simulation
ci: lint typecheck test-unit security-scan
	@echo "CI checks passed!"

# Database migrations
migrate-up:
	uv run python scripts/migration/migrate.py up

migrate-down:
	uv run python scripts/migration/migrate.py down

# Model registry
mlflow-ui:
	docker-compose -f infra/docker-compose.yaml exec mlflow mlflow ui --host 0.0.0.0 --port 5000

# Feast
feast-apply:
	cd libs/feature_store && feast apply

feast-materialize:
	cd libs/feature_store && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Kubernetes
k8s-deploy:
	kubectl apply -k infra/k8s/overlays/dev

k8s-delete:
	kubectl delete -k infra/k8s/overlays/dev

k8s-logs:
	kubectl logs -l app=upi-fraud-detector -f