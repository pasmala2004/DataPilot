"""The 20 evaluation questions (10 per dataset) and their ground truth.

Each ``Question`` has plain-English ``text`` (what a user would type) and a ``truth(df)`` function that
computes the expected figures with plain pandas/scipy. Nothing is hard-coded: the truth is recomputed from
the data files every time. The rules mirror the tools: null cells are dropped, filters ignore null cells,
and a top-n keeps the n largest groups.

Datasets live in ``eval/data`` (gitignored): ``shopping_trends.csv`` (no date column, so no period question)
and ``SeoulBikeData.csv`` (Windows-1252, day/month/year dates).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent / "data"
SHOPPING = "shopping_trends.csv"
SEOUL = "SeoulBikeData.csv"
KINDS = (
    "scalar", "group", "topn", "filtered", "period", "correlation", "test", "two_task", "pandas", "unanswerable",
)


@dataclass(frozen=True)
class Expect:
    """One expected figure.

    ``field`` says where the tool reports it: ``value`` (aggregate scalar or group value), ``r``, ``p_value``,
    ``statistic``, ``group_means`` (``group`` is the label) or ``text`` (a number printed by run_pandas).
    ``group`` is the group label as text (``None`` for a single figure).
    """

    name: str
    value: float
    field: str = "value"
    group: Optional[str] = None


@dataclass(frozen=True)
class Question:
    """One evaluation question. ``truth`` is empty for an unanswerable question (zero facts expected)."""

    id: str
    dataset: str
    sheet: str
    text: str
    kind: str
    truth: Callable[[pd.DataFrame], list[Expect]]
    unanswerable: bool = False
    expects_chart: bool = False


# --------------------------------------------------------------------------- #
# Loading (plain pandas, independent of core.py)
# --------------------------------------------------------------------------- #
def read_dataset(file: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read an eval dataset with plain pandas (encoding fallback; day-first dates for SeoulBikeData)."""
    path = data_dir / file
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - cp1252 can decode almost anything
        df = pd.read_csv(path, encoding="latin-1")
    for col in df.columns:
        if "date" in str(col).lower() and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], dayfirst=True)
    return df


# --------------------------------------------------------------------------- #
# Truth helpers (rules of the tools: drop nulls first)
# --------------------------------------------------------------------------- #
def _scalar(name: str, value: float) -> list[Expect]:
    return [Expect(name, float(value))]


def _groups(name: str, series: pd.Series) -> list[Expect]:
    return [Expect(f"{name} [{label}]", float(v), group=_label(label)) for label, v in series.items()]


def _label(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _corr(x: str, y: str, df: pd.DataFrame) -> list[Expect]:
    both = df[[x, y]].dropna()
    r, p = stats.pearsonr(both[x], both[y])
    return [Expect(f"r({x},{y})", float(both[x].corr(both[y])), field="r"), Expect(f"p({x},{y})", float(p), field="p_value")]


def _ttest(x: str, y: str, df: pd.DataFrame) -> list[Expect]:
    sub = df[[x, y]].dropna()
    samples = [g[y].to_numpy() for _, g in sub.groupby(x, sort=True)]
    res = stats.ttest_ind(samples[0], samples[1], equal_var=False)
    out = [Expect(f"t({y} by {x})", float(res.statistic), field="statistic"),
           Expect(f"p({y} by {x})", float(res.pvalue), field="p_value")]
    out += [Expect(f"mean {y} [{k}]", float(v), field="group_means", group=_label(k)) for k, v in sub.groupby(x)[y].mean().items()]
    return out


def _anova(x: str, y: str, df: pd.DataFrame) -> list[Expect]:
    sub = df[[x, y]].dropna()
    samples = [g[y].to_numpy() for _, g in sub.groupby(x, sort=True)]
    res = stats.f_oneway(*samples)
    out = [Expect(f"F({y} by {x})", float(res.statistic), field="statistic"),
           Expect(f"p({y} by {x})", float(res.pvalue), field="p_value")]
    out += [Expect(f"mean {y} [{k}]", float(v), field="group_means", group=_label(k)) for k, v in sub.groupby(x)[y].mean().items()]
    return out


# --------------------------------------------------------------------------- #
# Shopping trends (3,900 purchases, no date column)
# --------------------------------------------------------------------------- #
AMOUNT, RATING = "Purchase Amount (USD)", "Review Rating"


def _shop_sum(df: pd.DataFrame) -> list[Expect]:
    return _scalar("sum amount", df[AMOUNT].dropna().sum())


def _shop_mean(df: pd.DataFrame) -> list[Expect]:
    return _scalar("mean rating", df[RATING].dropna().mean())


def _shop_group(df: pd.DataFrame) -> list[Expect]:
    return _groups("sum amount by Category", df.dropna(subset=[AMOUNT, "Category"]).groupby("Category")[AMOUNT].sum())


def _shop_topn(df: pd.DataFrame) -> list[Expect]:
    sums = df.dropna(subset=[AMOUNT, "Location"]).groupby("Location")[AMOUNT].sum()
    return _groups("top locations", sums.sort_values(ascending=False, kind="stable").head(5))


def _shop_filtered(df: pd.DataFrame) -> list[Expect]:
    sub = df[(df["Subscription Status"] == "Yes") & (df["Age"] >= 50)]
    return _scalar("count subscribed 50+", sub["Customer ID"].dropna().count())


def _shop_corr(df: pd.DataFrame) -> list[Expect]:
    return _corr("Age", AMOUNT, df)


def _shop_ttest(df: pd.DataFrame) -> list[Expect]:
    return _ttest("Subscription Status", AMOUNT, df)


def _shop_two_task(df: pd.DataFrame) -> list[Expect]:
    return _groups("mean rating by Season", df.dropna(subset=[RATING, "Season"]).groupby("Season")[RATING].mean())


def _shop_pandas(df: pd.DataFrame) -> list[Expect]:
    promo = df["Promo Code Used"].dropna()
    return [Expect("percent promo code", float((promo == "Yes").mean() * 100), field="text")]


# --------------------------------------------------------------------------- #
# Seoul bike sharing (8,760 hourly rows, day/month/year dates)
# --------------------------------------------------------------------------- #
BIKES, TEMP = "Rented Bike Count", "Temperature(°C)"


def _bike_sum(df: pd.DataFrame) -> list[Expect]:
    return _scalar("sum bikes", df[BIKES].dropna().sum())


def _bike_group(df: pd.DataFrame) -> list[Expect]:
    return _groups("mean bikes by Holiday", df.dropna(subset=[BIKES, "Holiday"]).groupby("Holiday")[BIKES].mean())


def _bike_topn(df: pd.DataFrame) -> list[Expect]:
    sums = df.dropna(subset=[BIKES, "Hour"]).groupby("Hour")[BIKES].sum()
    return _groups("top hours", sums.sort_values(ascending=False, kind="stable").head(3))


def _bike_filtered(df: pd.DataFrame) -> list[Expect]:
    sub = df[(df["Holiday"] == "Holiday") & (df["Functioning Day"] == "Yes")]
    return _scalar("mean bikes holiday functioning", sub[BIKES].dropna().mean())


def _bike_period(df: pd.DataFrame) -> list[Expect]:
    sub = df[(df["Date"] >= "2018-06-01") & (df["Date"] <= "2018-06-30")]
    return _scalar("sum bikes June 2018", sub[BIKES].dropna().sum())


def _bike_corr(df: pd.DataFrame) -> list[Expect]:
    return _corr(TEMP, BIKES, df)


def _bike_anova(df: pd.DataFrame) -> list[Expect]:
    return _anova("Seasons", BIKES, df)


def _bike_two_task(df: pd.DataFrame) -> list[Expect]:
    return _groups("sum bikes by Seasons", df.dropna(subset=[BIKES, "Seasons"]).groupby("Seasons")[BIKES].sum())


def _bike_pandas(df: pd.DataFrame) -> list[Expect]:
    return [Expect("p90 bikes", float(df[BIKES].dropna().quantile(0.9)), field="text")]


def _none(df: pd.DataFrame) -> list[Expect]:
    return []


QUESTIONS: tuple[Question, ...] = (
    Question("shop_01_sum", SHOPPING, "csv", "What is the total purchase amount across all purchases?", "scalar", _shop_sum),
    Question("shop_02_mean", SHOPPING, "csv", "What is the average review rating?", "scalar", _shop_mean),
    Question("shop_03_group", SHOPPING, "csv", "Show the total purchase amount for each category.", "group", _shop_group),
    Question("shop_04_topn", SHOPPING, "csv", "Which 5 locations have the highest total purchase amount?", "topn", _shop_topn),
    Question("shop_05_filtered", SHOPPING, "csv",
             "How many purchases were made by customers with a subscription who are 50 years old or older?", "filtered", _shop_filtered),
    Question("shop_06_corr", SHOPPING, "csv", "Is there a correlation between customer age and purchase amount?", "correlation", _shop_corr),
    Question("shop_07_ttest", SHOPPING, "csv",
             "Do subscribers and non-subscribers differ in purchase amount? Run a t-test.", "test", _shop_ttest),
    Question("shop_08_chart", SHOPPING, "csv",
             "Give me the average review rating for each season and draw a bar chart of it.", "two_task", _shop_two_task, expects_chart=True),
    Question("shop_09_pandas", SHOPPING, "csv",
             "What percentage of purchases used a promo code?", "pandas", _shop_pandas),
    Question("shop_10_unanswerable", SHOPPING, "csv", "What was the profit margin on each item?", "unanswerable", _none, unanswerable=True),
    Question("bike_01_sum", SEOUL, "csv", "What is the total number of rented bikes?", "scalar", _bike_sum),
    Question("bike_02_group", SEOUL, "csv", "What is the average number of rented bikes on holidays versus non-holidays?", "group", _bike_group),
    Question("bike_03_topn", SEOUL, "csv", "Which 3 hours of the day have the most rented bikes in total?", "topn", _bike_topn),
    Question("bike_04_filtered", SEOUL, "csv",
             "What is the average number of rented bikes per hour on holidays when the system was functioning?", "filtered", _bike_filtered),
    Question("bike_05_period", SEOUL, "csv",
             "How many bikes were rented in total between 2018-06-01 and 2018-06-30?", "period", _bike_period),
    Question("bike_06_corr", SEOUL, "csv", "How strongly does temperature correlate with the number of rented bikes?", "correlation", _bike_corr),
    Question("bike_07_anova", SEOUL, "csv", "Does the number of rented bikes differ between seasons? Run an ANOVA.", "test", _bike_anova),
    Question("bike_08_chart", SEOUL, "csv",
             "Show the total rented bikes for each season and plot it as a bar chart.", "two_task", _bike_two_task, expects_chart=True),
    Question("bike_09_pandas", SEOUL, "csv", "What is the 90th percentile of the hourly rented bike count?", "pandas", _bike_pandas),
    Question("bike_10_unanswerable", SEOUL, "csv", "Which bike station had the most rentals?", "unanswerable", _none, unanswerable=True),
)


def get_question(question_id: str) -> Question:
    """The question with this id (``KeyError`` if there is none)."""
    for q in QUESTIONS:
        if q.id == question_id:
            return q
    raise KeyError(question_id)
