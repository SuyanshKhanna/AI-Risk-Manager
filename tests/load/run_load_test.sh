#!/bin/bash
# Load test runner for UPI Fraud Detector API
# Usage: ./run_load_test.sh [mode] [host]
# Modes: constant, step, spike
# Default: constant load at 100 RPS for 5 minutes

set -e

MODE=${1:-constant}
HOST=${2:-http://localhost:8000}
LOCUST_FILE="tests/load/locustfile.py"

echo "=== UPI Fraud Detector Load Test ==="
echo "Mode: $MODE"
echo "Target: $HOST"
echo "Locust file: $LOCUST_FILE"
echo ""

# Check if locust is installed
if ! command -v locust &> /dev/null; then
    echo "Installing locust..."
    pip install locust>=2.15
fi

# Check if service is healthy
echo "Checking service health..."
if ! curl -sf "$HOST/health" > /dev/null; then
    echo "ERROR: Service not healthy at $HOST"
    echo "Start the service first: python -m services.upi_fraud_detector.serve --model models/upi_fraud_lgbm --port 8000"
    exit 1
fi
echo "Service is healthy"

case $MODE in
    constant)
        echo "Running constant load: 100 RPS for 5 minutes"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users=100 \
            --spawn-rate=20 \
            --run-time=300s \
            --headless \
            --html=load_test_report.html \
            --csv=load_test_results
        ;;
    step)
        echo "Running step load: ramp 10->50->100->50->10 RPS"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --class-picker \
            --headless \
            --html=load_test_report.html \
            --csv=load_test_results
        ;;
    spike)
        echo "Running spike test: burst traffic"
        locust -f "$LOCUST_FILE" \
            --host="$HOST" \
            --users=200 \
            --spawn-rate=50 \
            --run-time=60s \
            --headless \
            --html=load_test_report.html \
            --csv=load_test_results
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [constant|step|spike] [host]"
        exit 1
        ;;
esac

echo ""
echo "=== Load Test Complete ==="
echo "Report: load_test_report.html"
echo "CSV results: load_test_results_*.csv"