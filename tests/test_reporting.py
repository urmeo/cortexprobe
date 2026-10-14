"""Verify recorded measurements, provenance, and figure coordinates."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from cortexprobe.config import RunConfig
from cortexprobe.geometry import Grid

from .conftest import load_script

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def report_script():
    return load_script("generate_validation_report.py")


@pytest.fixture
def record():
    return json.loads((ROOT / "results" / "validation.json").read_text())


def marked_document(script, record):
    empty = "\n".join(
        f"<!-- BEGIN GENERATED: {name} -->\n<!-- END GENERATED: {name} -->"
        for name in ("recovery", "uncertainty", "cross_validation", "runtime")
    )
    return script.splice(empty, script.render(record))


def prepare_run(script, record, monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "validation.json").write_text(json.dumps(record))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "validation.md").write_text(marked_document(script, record))
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "RESULTS", results)
    for section in ("recovery", "uncertainty", "cross_validation"):
        monkeypatch.setattr(
            script, f"measure_{section}", lambda *args, name=section: copy.deepcopy(record[name])
        )
    environment = {key: f"current-{key}" for key in record["environment"]}
    monkeypatch.setattr(script, "current_environment", lambda: environment)
    return results / "validation.json", environment


def test_report_loads_actual_json_configuration(monkeypatch):
    original = RunConfig.load(ROOT / "configs" / "validation.json")
    changed = replace(original, seed=123)
    loaded_paths = []

    def load(path):
        loaded_paths.append(path)
        return changed

    monkeypatch.setattr(RunConfig, "load", staticmethod(load))
    script = load_script("generate_validation_report.py")
    assert loaded_paths == [ROOT / "configs" / "validation.json"]
    assert script.CONFIG.seed == 123
    assert script.CONFIG.digest() == changed.digest()


def test_generated_blocks_use_recorded_counts_and_paired_coverage(report_script, record):
    record["recovery"]["pure_noise"]["rejected"] = 37
    blocks = report_script.render(record)
    assert set(blocks) == {"recovery", "uncertainty", "cross_validation", "runtime"}
    assert "pure noise 3/40 accepted" in blocks["recovery"]
    assert "paired seeds across cells" in blocks["uncertainty"]
    combined = "\n".join(blocks.values())
    assert "4000" not in combined
    assert "50 fresh" not in combined
    assert "pooled" not in combined
    assert all(block.count("<sub>") == 1 for block in blocks.values())
    assert "figure pixels" not in blocks["runtime"]


def test_check_accepts_tolerated_measurements_but_rejects_drift(report_script, record, capsys):
    document = marked_document(report_script, record)
    fresh = copy.deepcopy(record)
    fresh["recovery"]["rows"][1]["position_mean"] += 0.000001
    assert report_script.run_check(fresh, record, document) == 0
    assert "within tolerance" in capsys.readouterr().out
    fresh["recovery"]["rows"][1]["position_mean"] += 0.01
    assert report_script.run_check(fresh, record, document) == 1
    assert "recovery.rows.1.position_mean" in capsys.readouterr().out


def test_check_rejects_a_manually_edited_validation_table(report_script, record, capsys):
    document = marked_document(report_script, record)
    changed = document.replace("| 0.2 |", "| 0.3 |", 1)
    assert changed != document
    assert report_script.run_check(record, record, changed) == 1
    assert "tables do not match" in capsys.readouterr().out


@pytest.mark.parametrize("replacement", [None, "invalid", [], {}, 10**400])
@pytest.mark.parametrize("changed_report", ["fresh", "recorded"])
def test_check_reports_invalid_measurement_drift(
    report_script, record, capsys, replacement, changed_report
):
    document = marked_document(report_script, record)
    fresh = copy.deepcopy(record)
    changed = fresh if changed_report == "fresh" else record
    changed["recovery"]["rows"][1]["position_mean"] = replacement
    assert report_script.run_check(fresh, record, document) == 1
    assert "recovery.rows.1.position_mean" in capsys.readouterr().out


def test_check_ignores_separately_recorded_hardware_provenance(report_script, record):
    fresh = copy.deepcopy(record)
    fresh.update(
        runtime={"rows": []},
        runtime_environment={"platform": "other"},
        runtime_generated="2026-10-06",
        runtime_config_digest="other-config",
    )
    assert report_script.run_check(fresh, record, marked_document(report_script, record)) == 0


def test_report_rejects_conflicting_check_and_runtime_flags(report_script, capsys):
    with pytest.raises(SystemExit) as error:
        report_script.main(["--runtime", "--check"])
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_tolerated_regeneration_preserves_original_record(
    report_script, record, monkeypatch, tmp_path
):
    path, _ = prepare_run(report_script, record, monkeypatch, tmp_path)
    recovery = copy.deepcopy(record["recovery"])
    recovery["rows"][1]["position_mean"] += 0.0000008
    monkeypatch.setattr(report_script, "measure_recovery", lambda *args: recovery)
    assert report_script.main([]) == 0
    assert json.loads(path.read_text()) == record


@pytest.mark.skipif(sys.platform == "win32", reason="The C locale selects ASCII on POSIX.")
def test_report_checks_and_rewrites_utf8_in_ascii_locale(
    report_script, record, monkeypatch, tmp_path
):
    path, _ = prepare_run(report_script, record, monkeypatch, tmp_path)
    document = tmp_path / "docs" / "validation.md"
    original_report = path.read_bytes()
    original_document = document.read_bytes()
    command = """
import importlib.util
import json
import locale
import sys
from pathlib import Path

assert locale.getpreferredencoding(False).lower() in ("ascii", "us-ascii", "ansi_x3.4-1968")
script_path, root = map(Path, sys.argv[1:])
sys.path.insert(0, str(script_path.parent))
spec = importlib.util.spec_from_file_location("report", script_path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
report.ROOT = root
report.RESULTS = root / "results"
record = json.loads((report.RESULTS / "validation.json").read_text(encoding="utf-8"))
for section in ("recovery", "uncertainty", "cross_validation"):
    setattr(report, f"measure_{section}", lambda *args, name=section: record[name])
assert report.main(["--check"]) == 0
assert report.main([]) == 0
"""
    environment = dict(os.environ, LC_ALL="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            str(ROOT / "scripts/generate_validation_report.py"),
            str(tmp_path),
        ],
        env=environment,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert json.loads(path.read_bytes()) == json.loads(original_report)
    assert document.read_bytes() == original_document


@pytest.mark.parametrize("separate_runtime", [False, True])
def test_changed_config_preserves_old_runtime_provenance(
    report_script, record, monkeypatch, tmp_path, separate_runtime
):
    if not separate_runtime:
        for key in ("runtime_environment", "runtime_generated", "runtime_config_digest"):
            record.pop(key, None)
    path, environment = prepare_run(report_script, record, monkeypatch, tmp_path)
    monkeypatch.setattr(report_script, "CONFIG", replace(report_script.CONFIG, seed=123))
    assert report_script.main([]) == 0
    updated = json.loads(path.read_text())
    assert updated["environment"] == environment
    assert updated["generated"] == date.today().isoformat()
    assert updated["config_digest"] != record["config_digest"]
    assert updated["runtime"] == record["runtime"]
    assert updated["runtime_environment"] == record.get(
        "runtime_environment", record["environment"]
    )
    assert updated["runtime_generated"] == record.get("runtime_generated", record["generated"])
    assert updated["runtime_config_digest"] == record.get(
        "runtime_config_digest", record["config_digest"]
    )
    runtime_block = report_script.render(updated)["runtime"]
    assert record["config_digest"][:12] in runtime_block
    assert updated["config_digest"][:12] not in runtime_block


def test_runtime_refresh_records_its_own_environment(report_script, record, monkeypatch, tmp_path):
    path, environment = prepare_run(report_script, record, monkeypatch, tmp_path)
    new_runtime = copy.deepcopy(record["runtime"])
    new_runtime["rows"][0]["fit_seconds"] += 1
    monkeypatch.setattr(report_script, "measure_runtime", lambda *args: new_runtime)
    assert report_script.main(["--runtime"]) == 0
    updated = json.loads(path.read_text())
    assert updated["environment"] == record["environment"]
    assert updated["generated"] == record["generated"]
    assert updated["runtime"] == new_runtime
    assert updated["runtime_environment"] == environment
    assert updated["runtime_generated"] == date.today().isoformat()
    assert updated["runtime_config_digest"] == record["config_digest"]


@pytest.fixture
def figures_script():
    pytest.importorskip("matplotlib")
    return load_script("generate_figures.py")


@pytest.mark.parametrize("contents", [None, "{", '{"config_digest": "stale"}'])
def test_figures_reject_missing_malformed_or_stale_records(
    figures_script, monkeypatch, tmp_path, contents, capsys
):
    monkeypatch.setattr(figures_script, "RESULTS", tmp_path)
    figures_path = tmp_path / "not-created"
    monkeypatch.setattr(figures_script, "FIGURES", figures_path)
    if contents is not None:
        (tmp_path / "validation.json").write_text(contents)
    assert figures_script.main([]) == 1
    assert "generate" in capsys.readouterr().err
    assert not figures_path.exists()


def test_figures_help_and_invalid_flags_do_not_generate_files(
    figures_script, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(figures_script, "RESULTS", tmp_path)
    with pytest.raises(SystemExit) as help_result:
        figures_script.main(["--help"])
    assert help_result.value.code == 0
    assert "11 scientific plots" in capsys.readouterr().out
    with pytest.raises(SystemExit) as invalid_result:
        figures_script.main(["--unknown"])
    assert invalid_result.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_figures_accept_current_record(figures_script, record, monkeypatch, tmp_path):
    monkeypatch.setattr(figures_script, "RESULTS", tmp_path)
    (tmp_path / "validation.json").write_text(json.dumps(record))
    assert figures_script.load_recorded_report() == record


@pytest.mark.parametrize(
    ("x0", "y0", "expected"),
    [(0.0, 0.0, (31.5, 31.5)), (12.0, 8.0, (43.5, 23.5)), (-12.0, -8.0, (19.5, 39.5))],
)
def test_field_overlay_uses_pixel_center_and_upward_y(figures_script, x0, y0, expected):
    assert figures_script.pixel_coordinates(Grid(64), x0, y0) == expected
