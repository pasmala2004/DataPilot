"""CSV encodings (UTF-8, BOM, Windows-1252) and day-first dates."""

from __future__ import annotations

from pathlib import Path

import core


def _cfg(tmp_path: Path) -> core.Config:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return core.Config(data_dir=data, output_dir=tmp_path / "out")


def test_cp1252_csv_with_degree_sign_loads(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "w.csv").write_bytes("Date,Temperature(°C),Count\n01/12/2017,1.5,10\n02/12/2017,2.5,20\n".encode("cp1252"))
    store: core.Store = {}
    assert core.list_sheets("w.csv", config=cfg)["sheets"][0]["rows"] == 2
    result = core.load_data("w.csv", store=store, config=cfg)
    assert "error" not in result
    assert "Temperature(°C)" in result["columns"]


def test_utf8_bom_csv_has_clean_first_header(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "b.csv").write_bytes(b"\xef\xbb\xbfname,v\na,1\n")
    result = core.load_data("b.csv", store={}, config=cfg)
    assert list(result["columns"]) == ["name", "v"]


def test_utf8_csv_still_works(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "u.csv").write_text("city,v\nالقاهرة,1\n", encoding="utf-8")
    assert "error" not in core.load_data("u.csv", store={}, config=cfg)


def test_day_first_dates_parse_with_info_warning(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "d.csv").write_text("Date,v\n01/12/2017,1\n13/12/2017,2\n", encoding="utf-8")
    result = core.load_data("d.csv", store={}, config=cfg)
    assert "datetime" in str(result["columns"]["Date"])
    assert result["date_ranges"]["Date"][0].startswith("2017-12-01")
    assert any("day/month/year" in w for w in result["warnings"])


def test_month_first_dates_unchanged_without_warning(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "m.csv").write_text("Date,v\n12/13/2017,1\n01/02/2018,2\n", encoding="utf-8")
    result = core.load_data("m.csv", store={}, config=cfg)
    assert "warnings" not in result
    assert result["date_ranges"]["Date"][0].startswith("2017-12-13")


def test_unparseable_dates_still_warn(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    (cfg.data_dir / "x.csv").write_text("Date,v\nhello,1\nworld,2\n", encoding="utf-8")
    result = core.load_data("x.csv", store={}, config=cfg)
    assert any("could not be parsed" in w for w in result["warnings"])
