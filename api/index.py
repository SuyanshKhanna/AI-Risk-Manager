import sys
from pathlib import Path

# Add root directory to sys.path so modules can be imported
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from services.upi_fraud_detector.serve import app
