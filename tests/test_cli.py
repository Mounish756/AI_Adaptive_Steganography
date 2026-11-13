"""
CLI Tests for main.py entry point.
Verifies embed, extract, analyze, and authentication error handling via CLI.
"""

import os
import sys
import subprocess
import pytest

PYTHON_EXE = sys.executable


def test_cli_help():
    """Verify main.py help message displays available commands."""
    res = subprocess.run([PYTHON_EXE, "main.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "embed" in res.stdout
    assert "extract" in res.stdout
    assert "analyze" in res.stdout
    assert "benchmark" in res.stdout


def test_cli_embed_and_extract_roundtrip(tmp_path):
    """Verify end-to-end embed and extract execution via CLI."""
    cover_path = "datasets/coco_subset/nat_fruits.jpg"
    stego_path = str(tmp_path / "cli_stego.png")
    secret_msg = "Automated CLI Integration Secret 2026"
    password = "CliSecretKey_2026!"

    # 1. Embed
    cmd_embed = [
        PYTHON_EXE, "main.py", "embed",
        "--cover", cover_path,
        "--message", secret_msg,
        "--password", password,
        "--output", stego_path
    ]
    res_embed = subprocess.run(cmd_embed, capture_output=True, text=True)
    assert res_embed.returncode == 0, f"Embed failed: {res_embed.stderr}"
    assert os.path.exists(stego_path)

    # 2. Extract with correct password
    cmd_extract = [
        PYTHON_EXE, "main.py", "extract",
        "--stego", stego_path,
        "--password", password
    ]
    res_extract = subprocess.run(cmd_extract, capture_output=True, text=True)
    assert res_extract.returncode == 0, f"Extract failed: {res_extract.stderr}"
    assert secret_msg in res_extract.stdout

    # 3. Extract with wrong password (must fail closed)
    cmd_wrong = [
        PYTHON_EXE, "main.py", "extract",
        "--stego", stego_path,
        "--password", "WrongKey#123"
    ]
    res_wrong = subprocess.run(cmd_wrong, capture_output=True, text=True)
    assert res_wrong.returncode != 0
    assert "AUTHENTICATION FAILED" in res_wrong.stdout


def test_cli_analyze(tmp_path):
    """Verify analyze command outputs suitability map."""
    cover_path = "datasets/coco_subset/nat_fruits.jpg"
    suit_path = str(tmp_path / "cli_suitability.png")

    cmd = [
        PYTHON_EXE, "main.py", "analyze",
        "--image", cover_path,
        "--output", suit_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists(suit_path)
