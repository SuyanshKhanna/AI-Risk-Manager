@echo off
REM Load test runner for UPI Fraud Detector API (Windows)
REM Usage: run_load_test.bat [mode] [host]
REM Modes: constant, step, spike
REM Default: constant load at 100 RPS for 5 minutes

set MODE=%1
if "%MODE%"=="" set MODE=constant

set HOST=%2
if "%HOST%"=="" set HOST=http://localhost:8000

set LOCUST_FILE=tests\load\locustfile.py

echo === UPI Fraud Detector Load Test ===
echo Mode: %MODE%
echo Target: %HOST%
echo Locust file: %LOCUST_FILE%
echo.

REM Check if locust is installed
where locust >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Installing locust...
    python -m pip install locust^>=2.15
)

REM Check if service is healthy
echo Checking service health...
curl -sf %HOST%/health >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Service not healthy at %HOST%
    echo Start the service first: python -m services.upi_fraud_detector.serve --model models/upi_fraud_lgbm --port 8000
    exit /b 1
)
echo Service is healthy

if /I "%MODE%"=="constant" (
    echo Running constant load: 100 RPS for 5 minutes
    locust -f "%LOCUST_FILE%" --host="%HOST%" --users=100 --spawn-rate=20 --run-time=300s --headless --html=load_test_report.html --csv=load_test_results
) else if /I "%MODE%"=="step" (
    echo Running step load: ramp 10-50-100-50-10 RPS
    locust -f "%LOCUST_FILE%" --host="%HOST%" --class-picker --headless --html=load_test_report.html --csv=load_test_results
) else if /I "%MODE%"=="spike" (
    echo Running spike test: burst traffic
    locust -f "%LOCUST_FILE%" --host="%HOST%" --users=200 --spawn-rate=50 --run-time=60s --headless --html=load_test_report.html --csv=load_test_results
) else (
    echo Unknown mode: %MODE%
    echo Usage: %0 [constant^|step^|spike] [host]
    exit /b 1
)

echo.
echo === Load Test Complete ===
echo Report: load_test_report.html
echo CSV results: load_test_results_*.csv