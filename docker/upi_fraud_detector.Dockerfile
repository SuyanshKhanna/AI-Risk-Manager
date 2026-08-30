# UPI Fraud Detector Service Dockerfile
FROM ai-risk-manager/base:latest AS upi-fraud-detector

# Copy service code
COPY --chown=apprunner:apprunner services/upi_fraud_detector/ ./services/upi_fraud_detector/

# Expose ports
EXPOSE 50051 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run service
CMD ["python", "-m", "upi_fraud_detector.serve"]