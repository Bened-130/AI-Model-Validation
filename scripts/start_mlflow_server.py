#!/usr/bin/env python
"""
Start local MLflow tracking server
Usage: python scripts/start_mlflow_server.py
"""

import subprocess
import os
from pathlib import Path

def main():
    # Create mlruns directory if not exists
    mlruns_dir = Path(__file__).parent.parent / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    
    # Start MLflow UI
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", f"sqlite:///{mlruns_dir}/mlflow.db",
        "--default-artifact-root", str(mlruns_dir),
        "--port", "5000"
    ]
    
    print(f"Starting MLflow server...")
    print(f"Tracking URI: http://localhost:5000")
    print(f"Artifacts: {mlruns_dir}")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")

if __name__ == "__main__":
    main()