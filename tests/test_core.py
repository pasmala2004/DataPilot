"""Tests for Task A: Config, safe_path, @traced, list_sheets, load_data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import core
from core import Config, list_sheets, load_data, safe_path, traced

OUTSIDE = {"error": "path outside allowed directory"}


def _digest(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


# ------------------------------------------------------------------ Config --
def test_config_defaults_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("OUTPUT_DIR", raising=False)
    assert Config().data_dir == Path("./data")
    assert Config().output_dir == Path("./outputs")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "o"))
    assert Config().data_dir == tmp_path / "d"
    assert Config().output_dir == tmp_path / "o"


# --------------------------------------------------------------- safe_path --
def test_safe_path_accepts_relative_and_absolute(cfg: Config, xlsx_path: Path) -> None:
    assert safe_path("book.xlsx", cfg.data_dir) == xlsx_path.resolve()
    assert safe_path(str(xlsx_path), cfg.data_dir) == xlsx_path.resolve()


def test_safe_path_rejects_dotdot(cfg: Config) -> None:
    assert safe_path("../secret.xlsx", cfg.data_dir) == OUTSIDE
    assert safe_path("sub/../../secret.xlsx", cfg.data_dir) == OUTSIDE


def test_safe_path_rejects_absolute_outside_and_sibling_prefix(cfg: Config) -> None:
    sibling = cfg.data_dir.parent / "data_evil"
    sibling.mkdir()
    assert safe_path(str(sibling / "x.xlsx"), cfg.data_dir) == OUTSIDE
    assert safe_path("/etc/passwd", cfg.data_dir) == OUTSIDE


def test_safe_path_rejects_symlink_escape(cfg: Config, tmp_path: Path) -> None:
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    link = cfg.data_dir / "link.csv"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not available")
    assert safe_path("link.csv", cfg.data_dir) == OUTSIDE


# ------------------------------------------------------------------ traced --
def test_traced_converts_exceptions_and_wrong_arguments() -> None:
    @traced
    def boom(x: int) -> dict:
        raise ValueError("bad thing")

    assert boom(1) == {"error": "ValueError: bad thing"}
    assert "TypeError" in boom()["error"]  # wrong arity must not raise either


# ------------------------------------------------------------- list_sheets --
def test_list_sheets_xlsx_matches_pandas(cfg: Config, xlsx_path: Path) -> None:
    result = list_sheets("book.xlsx", config=cfg)
    expected = pd.read_excel(xlsx_path, sheet_name=None)
    assert [s["name"] for s in result["sheets"]] == list(expected)
    for sheet in result["sheets"]:
        df = expected[sheet["name"]]
        assert sheet["rows"] == len(df)
        assert sheet["columns"] == df.shape[1]
    assert result["sheets"][0] == {"name": "Sales", "rows": 6, "columns": 4}


def test_list_sheets_csv_has_single_sheet_named_csv(cfg: Config, csv_path: Path) -> None:
    assert list_sheets("sales.csv", config=cfg) == {
        "sheets": [{"name": "csv", "rows": 6, "columns": 4}]
    }


@pytest.mark.parametrize("tool", ["list_sheets", "load_data"])
def test_bad_extension_missing_file_and_traversal(cfg: Config, tool: str) -> None:
    (cfg.data_dir / "notes.txt").write_text("hello")
    (cfg.data_dir / "old.xls").write_bytes(b"x")
    call = (lambda p: list_sheets(p, config=cfg)) if tool == "list_sheets" else (
        lambda p: load_data(p, store={}, config=cfg)
    )
    assert "unsupported file type '.txt'" in call("notes.txt")["error"]
    assert "unsupported file type '.xls'" in call("old.xls")["error"]
    assert call("missing.xlsx") == {"error": "file not found"}
    assert call("../outside.xlsx") == OUTSIDE


def test_corrupt_xlsx_returns_error_instead_of_raising(cfg: Config) -> None:
    (cfg.data_dir / "broken.xlsx").write_bytes(b"this is not a zip file")
    assert "error" in list_sheets("broken.xlsx", config=cfg)
    assert "error" in load_data("broken.xlsx", store={}, config=cfg)


# --------------------------------------------------------------- load_data --
def test_load_data_xlsx_matches_pandas(cfg: Config, xlsx_path: Path, store: dict) -> None:
    result = load_data("book.xlsx", "Sales", store=store, config=cfg)
    expected = pd.read_excel(xlsx_path, sheet_name="Sales")

    assert result["dataset_id"] == "ds1"
    assert result["rows"] == len(expected) == 6  # duplicate row is kept, not dropped
    assert result["columns"] == {c: str(t) for c, t in expected.dtypes.items()}
    assert result["columns"]["order_date"].startswith("datetime64")
    assert not result["columns"]["update_count"].startswith("datetime64")
    assert result["nulls"] == {"region": 1, "amount": 1}  # only n > 0
    assert result["date_ranges"] == {
        "order_date": ["2024-01-05T00:00:00", "2024-03-15T00:00:00"]
    }
    assert "warnings" not in result
    pd.testing.assert_frame_equal(store["ds1"], expected)


def test_load_data_preview_is_first_5_rows_and_json_safe(cfg: Config, xlsx_path: Path) -> None:
    result = load_data("book.xlsx", store={}, config=cfg)  # default = first sheet
    preview = result["preview"]
    assert len(preview) == 5
    assert preview[0] == {
        "order_date": "2024-01-05T00:00:00",
        "region": "North",
        "amount": 120.5,
        "update_count": 1,
    }
    assert preview[4]["amount"] is None  # NaN becomes null
    json.dumps(result, allow_nan=False)  # fully JSON-serialisable


def test_load_data_csv_parses_date_columns(cfg: Config, csv_path: Path, store: dict) -> None:
    result = load_data("sales.csv", store=store, config=cfg)
    expected = pd.read_csv(csv_path)
    assert result["rows"] == len(expected)
    assert result["nulls"] == {"region": 1, "amount": 1}
    assert result["columns"]["order_date"].startswith("datetime64")
    assert result["date_ranges"]["order_date"] == ["2024-01-05T00:00:00", "2024-03-15T00:00:00"]
    assert pd.api.types.is_datetime64_any_dtype(store["ds1"]["order_date"])


def test_load_data_second_sheet_and_incrementing_ids(cfg: Config, xlsx_path: Path, store: dict) -> None:
    first = load_data("book.xlsx", "Sales", store=store, config=cfg)
    second = load_data("book.xlsx", "Targets", store=store, config=cfg)
    assert (first["dataset_id"], second["dataset_id"]) == ("ds1", "ds2")
    assert second["rows"] == 3
    expected = pd.read_excel(xlsx_path, sheet_name="Targets")
    assert second["columns"] == {c: str(t) for c, t in expected.dtypes.items()}
    assert second["nulls"] == {} and second["date_ranges"] == {}
    assert set(store) == {"ds1", "ds2"}


def test_failed_load_does_not_consume_an_id(cfg: Config, xlsx_path: Path, store: dict) -> None:
    assert "error" in load_data("book.xlsx", "Nope", store=store, config=cfg)
    assert load_data("book.xlsx", "Sales", store=store, config=cfg)["dataset_id"] == "ds1"


def test_unknown_sheet_lists_available_sheets(cfg: Config, xlsx_path: Path, csv_path: Path) -> None:
    error = load_data("book.xlsx", "Nope", store={}, config=cfg)["error"]
    assert "Nope" in error and "Sales" in error and "Targets" in error
    assert "csv" in load_data("sales.csv", "Nope", store={}, config=cfg)["error"]
    assert "dataset_id" in load_data("sales.csv", "csv", store={}, config=cfg)


def test_row_limit(monkeypatch: pytest.MonkeyPatch, cfg: Config, xlsx_path: Path, csv_path: Path) -> None:
    monkeypatch.setattr(core, "MAX_ROWS", 5)  # the fixtures have 6 rows
    for name in ("book.xlsx", "sales.csv"):
        store: dict = {}
        assert "too many rows" in load_data(name, store=store, config=cfg)["error"]
        assert store == {}
    monkeypatch.setattr(core, "MAX_ROWS", 6)  # exactly at the limit is fine
    assert load_data("sales.csv", store={}, config=cfg)["rows"] == 6


def test_unparseable_date_column_is_left_alone_with_warning(cfg: Config) -> None:
    pd.DataFrame({"due_date": ["2024-01-01", "not a date"], "x": [1, 2]}).to_csv(
        cfg.data_dir / "bad_dates.csv", index=False
    )
    store: dict = {}
    result = load_data("bad_dates.csv", store=store, config=cfg)
    assert not result["columns"]["due_date"].startswith("datetime64")
    assert result["nulls"] == {}  # bad values were NOT silently turned into nulls
    assert "due_date" in result["warnings"][0]


# ---------------------------------------------------- safety-style checks ---
def test_input_files_are_never_modified(cfg: Config, xlsx_path: Path, csv_path: Path) -> None:
    before = {p: _digest(p) for p in (xlsx_path, csv_path)}
    listing_before = sorted(os.listdir(cfg.data_dir))
    for name in ("book.xlsx", "sales.csv"):
        list_sheets(name, config=cfg)
        load_data(name, store={}, config=cfg)
    assert {p: _digest(p) for p in before} == before
    assert sorted(os.listdir(cfg.data_dir)) == listing_before  # no extra files either


def test_stderr_logs_contain_names_and_sizes_but_no_cell_values(
    cfg: Config, capfd: pytest.CaptureFixture[str]
) -> None:
    pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2031-07-04", "2031-07-05"]),
            "region": ["Zanzibar-Q9", "Wakanda-X3"],
            "amount": [424242.42, 737373.73],
        }
    ).to_excel(cfg.data_dir / "confidential_file.xlsx", sheet_name="SecretSheet", index=False)

    list_sheets("confidential_file.xlsx", config=cfg)
    load_data("confidential_file.xlsx", "SecretSheet", store={}, config=cfg)
    load_data("confidential_file.xlsx", "WrongSheet", store={}, config=cfg)  # error path

    err = capfd.readouterr().err
    assert "tool=list_sheets" in err and "tool=load_data" in err
    assert "args=['path', 'config']" in err and "args=['path', 'sheet', 'store', 'config']" in err
    assert "result_bytes=" in err
    for value in (
        "Zanzibar-Q9", "Wakanda-X3", "424242", "737373", "2031-07", "SecretSheet",
        "WrongSheet", "confidential_file",
    ):
        assert value not in err, f"{value!r} leaked into stderr logs"


def test_nothing_is_written_to_stdout(cfg: Config, xlsx_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    list_sheets("book.xlsx", config=cfg)
    load_data("book.xlsx", store={}, config=cfg)
    load_data("../x.xlsx", store={}, config=cfg)
    assert capfd.readouterr().out == ""
