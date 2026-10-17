"""Shared fixtures: tiny files built with pandas inside a temp DATA_DIR.

No network, and nothing from the real ./data folder is ever touched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core import Config


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """6 rows: 1 null region, 1 null amount, rows 2 and 3 are exact duplicates."""
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-02-11", "2024-02-11", "2024-03-02", "2024-03-15"]
            ),
            "region": ["North", "South", "North", "North", "East", None],
            "amount": [120.5, 80.0, 200.0, 200.0, None, 55.25],
            # name contains "date" but it is numeric: must NOT become a datetime
            "update_count": [1, 2, 3, 3, 4, 5],
        }
    )


@pytest.fixture
def targets_df() -> pd.DataFrame:
    """Second sheet, no date column."""
    return pd.DataFrame({"region": ["North", "South", "East"], "target": [500, 300, 250]})


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Config pointing at a temp DATA_DIR / OUTPUT_DIR."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir()
    out_dir.mkdir()
    return Config(data_dir=data_dir, output_dir=out_dir)


@pytest.fixture
def xlsx_path(cfg: Config, sales_df: pd.DataFrame, targets_df: pd.DataFrame) -> Path:
    """Workbook with two sheets: Sales and Targets."""
    path = cfg.data_dir / "book.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sales_df.to_excel(writer, sheet_name="Sales", index=False)
        targets_df.to_excel(writer, sheet_name="Targets", index=False)
    return path


@pytest.fixture
def csv_path(cfg: Config, sales_df: pd.DataFrame) -> Path:
    """CSV version of the Sales sheet."""
    path = cfg.data_dir / "sales.csv"
    sales_df.to_csv(path, index=False)
    return path


@pytest.fixture
def store() -> dict[str, pd.DataFrame]:
    """Empty dataset store."""
    return {}


@pytest.fixture
def analysis_df() -> pd.DataFrame:
    """60 deterministic rows (Nov 2023 - Aug 2024) with nulls in amount/region/date.

    Row 0 is dated 2024-03-31 15:30 to exercise the inclusive end-of-day rule.
    """
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "order_date": pd.Timestamp("2023-11-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": rng.choice(["A", "B"], n),
            "amount": rng.normal(100, 25, n).round(2),
            "qty": rng.integers(1, 10, n),
        }
    )
    df.loc[[3, 17, 40], "amount"] = np.nan
    df.loc[[5, 22, 40], "region"] = None
    df.loc[8, "order_date"] = pd.NaT
    df.loc[0, "order_date"] = pd.Timestamp("2024-03-31 15:30")
    return df


@pytest.fixture
def analysis_store(analysis_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Store holding ``analysis_df`` as ``ds1``."""
    return {"ds1": analysis_df}


@pytest.fixture
def orders() -> pd.DataFrame:
    """10 orders; one null amount; a `customer` column of canary strings that must never leak."""
    return pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-11", "2024-02-18", "2024-03-02",
                                      "2024-03-15", "2024-04-01", "2024-04-09", "2024-04-20", "2024-05-02"]),
        "region": ["North", "South", "North", "East", "South", "North", "East", "South", "North", "East"],
        "amount": [120.5, 80.0, 200.0, 95.25, 60.0, 150.75, 70.0, 99.5, 130.0, None],
        "qty": [3, 2, 5, 2, 1, 4, 2, 3, 4, 1],
        "customer": [f"Acme-{7000 + i}-Corp" for i in range(10)],
    })


@pytest.fixture
def workspace(tmp_path: Path, orders: pd.DataFrame) -> dict[str, Any]:
    data_dir, out_dir = tmp_path / "data", tmp_path / "outputs"
    data_dir.mkdir()
    out_dir.mkdir()
    with pd.ExcelWriter(data_dir / "sales.xlsx", engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="orders", index=False)
        pd.DataFrame({"region": ["North", "South", "East"], "target": [500, 300, 250]}).to_excel(
            writer, sheet_name="targets", index=False)
    return {"data_dir": data_dir, "out_dir": out_dir, "orders": orders}
