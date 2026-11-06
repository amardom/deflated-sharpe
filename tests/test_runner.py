import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from deflated_sharpe import summarize

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("output", ["table", "json", "csv"])
def test_user_input_to_output(output):
    result = subprocess.run([sys.executable, str(ROOT / "examples/run.py"), "--format", output],
                            capture_output=True, text=True, check=True, cwd=ROOT.parent)
    expected = summarize(**json.loads((ROOT / "examples/input.json").read_text()))
    if output == "json":
        assert json.loads(result.stdout) == expected
    elif output == "csv":
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert len(rows) == len(expected)
        for actual, row in zip(rows, expected):
            assert float(actual["dsr"]) == row["dsr"]
            assert actual["trial"] == row["trial"]
    else:
        assert "dsr" in result.stdout
        assert "variant_c" in result.stdout
        assert f"{expected[0]['dsr']:.6f}" in result.stdout

@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"returns": [[1,2]]}'])
def test_invalid_input_file(tmp_path, contents):
    source = tmp_path / "input.json"
    source.write_text(contents)
    result = subprocess.run([sys.executable, str(ROOT / "examples/run.py"), str(source)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr

def test_missing_input_file(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "examples/run.py"), str(tmp_path / "missing.json")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr

def test_console_retains_every_field_without_repeating_search_metadata():
    result = subprocess.run([sys.executable,str(ROOT/"examples/run.py")], capture_output=True,text=True,check=True)
    expected = summarize(**json.loads((ROOT/"examples/input.json").read_text()))
    for key in expected[0]:
        assert key in result.stdout
    assert result.stdout.count("estimated_independent_trials") == 1
    assert "Search inputs and estimates" in result.stdout
    assert "Trial results" in result.stdout
