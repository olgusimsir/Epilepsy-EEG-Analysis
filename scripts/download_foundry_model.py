"""Download + warm up a Foundry Local model and run a live test through our provider.

First run downloads the execution providers and the model (a few GB) — progress
prints to stderr. After that it's fully offline and cached.

Usage:
    ./venv/bin/python scripts/download_foundry_model.py                # phi-3.5-mini
    ./venv/bin/python scripts/download_foundry_model.py qwen2.5-0.5b   # small/fast, good first test
    ./venv/bin/python scripts/download_foundry_model.py phi-4
"""
import os
import sys

# Make the project root importable when this file is run directly (not via -m).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.foundry_local import FoundryLocalProvider, DEFAULT_MODEL_ALIAS


def main():
    alias = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ALIAS
    print(f"Downloading + loading Foundry Local model '{alias}'.")
    print("First run pulls a few GB (cached afterwards); progress prints below.\n")

    provider = FoundryLocalProvider(model_alias=alias)
    output = provider.generate(
        "You are a clinical neurophysiology assistant.",
        "Reply with ONE short sentence confirming you are running locally and offline.",
    )

    print("\n\n=== MODEL OUTPUT ===")
    print(output)
    print("\n✓ Foundry Local works.")
    print(f"  Use it for reports with:  REPORT_LLM_PROVIDER=foundry_local REPORT_LLM_MODEL={alias}")


if __name__ == "__main__":
    main()