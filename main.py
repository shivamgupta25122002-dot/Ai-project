import sys
import os

# Ensure model dependencies are verified on startup
from models.download_models import ensure_models_downloaded
from cli import run_cli

if __name__ == "__main__":
    ensure_models_downloaded()
    run_cli()
