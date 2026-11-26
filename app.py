from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MRK Equity Research", page_icon="📊", layout="wide")

# ----------------------------- Configuration -----------------------------
SEC_HEADERS = {
    "User-Agent": "MRK-Equity-Research-Student student.project@example.com"
}
SEC_CIK = "0000310158"
FRED_DEFAULT_SERIES = {
    "10-Year Treasury": "DGS10",
    "Effective Fed Funds Rate": "FEDFUNDS",
    "CPI": "CPIAUCSL",
}
PEERS = ["PFE", "ABBV", "BMY"]
PEER_NAMES = {"MRK": "Merck & Co.", "PFE": "Pfizer", "ABBV": "AbbVie", "BMY": "Bristol-Myers Squibb"}
TICKER = "MRK"
PRICE_YEARS = 10  # price-history window; all price labels below use this value

st.title("Merck & Co. (MRK) | Equity Research & Valuation")
st.caption("Educational research tool. Not investment advice. Market and filing data may be delayed or revised.")

with st.sidebar:
    st.header("Data & assumptions")
    st.markdown("**Data sources**")
    st.write("Yahoo Finance · SEC EDGAR · FRED")
    if st.button("Clear cached data / retry"):
        st.cache_data.clear()
        st.rerun()
    fred_key = st.text_input("FRED API key (optional)", type="password",
                             help="Get a key from fred.stlouisfed.org. Without it, enter the risk-free rate manually.")
    st.markdown("---")
    as_of = st.date_input("Valuation date", value=date.today())
    st.markdown("---")
    st.subheader("DCF assumptions")
    forecast_years = st.slider("Forecast horizon (years)", 5, 10, 5)
    risk_free_manual = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=15.0, value=4.2, step=0.1)
    equity_risk_premium = st.number_input("Equity risk premium (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    cost_of_debt = st.number_input("Pre-tax cost of debt (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
    tax_rate = st.slider("Marginal tax rate (%)", 0, 40, 15)
    terminal_growth = st.number_input("Terminal growth rate (%)", min_value=0.0, max_value=6.0, value=2.5, step=0.1)
    st.markdown("---")
    st.subheader("Patent-cliff assumptions")
    keytruda_revenue = st.number_input(
        "Latest annual Keytruda revenue ($bn)", min_value=0.0, value=29.5, step=0.5,
        help="Replace this illustrative starting value with the latest reported figure from Merck's annual report.")
    non_keytruda_growth = st.number_input("Non-Keytruda annual revenue growth (%)", min_value=-10.0, max_value=20.0, value=5.0, step=0.5)
    pre_cliff_growth = st.number_input("Keytruda growth before erosion (%)", min_value=-20.0, max_value=30.0, value=5.0, step=0.5)
    erosion = st.number_input("Annual Keytruda erosion after cliff begins (%)", min_value=0.0, max_value=80.0, value=25.0, step=1.0)
    cliff_start_year = st.number_input("First erosion year", min_value=2026, max_value=2035, value=2029, step=1)
    st.caption("Patent timing and biosimilar entry are uncertain. Treat the erosion rate and start year as scenarios, not facts.")


# ----------------------------- Generic helpers -----------------------------
def _num(x):
    """Convert to float or NaN."""
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _first_valid(*vals):
    for v in vals:
        v = _num(v)
        if pd.notna(v) and v != 0:
            return v
    return np.nan


def fmt_bn(x):
    return "—" if pd.isna(x) else f"${x:,.2f}B"


def safe_latest(series, default=np.nan):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def latest_row(df, cols):
    """Latest fiscal year where ALL requested columns have data (keeps years consistent)."""
    if df is None or df.empty:
        return None
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    sub = df.dropna(subset=cols)
    return sub.iloc[-1] if not sub.empty else None


def show_df(df, fmt="{:,.2f}"):
    """Display a dataframe with missing values shown as an em dash."""
    if df is None or df.empty:
        st.info("No data available.")
        return

    def _fmt(v):
        if pd.isna(v):
            return "—"
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)

    st.dataframe(df.style.format(_fmt))


# ----------------------------- Yahoo Finance -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_history(symbol, period="5y"):
    # Raises on failure -> failures are NOT cached.
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True, actions=True)
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo returned no price history for {symbol}")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_statements(symbol):
    t = yf.Ticker(symbol)
    return {"income": t.financials, "balance": t.balance_sheet, "cashflow": t.cashflow}


@st.cache_data(ttl=3600, show_spinner=False)
def _yahoo_snapshot_cached(symbol):
    """
    Build a market snapshot from several Yahoo endpoints with fallbacks.
    Raises if nothing useful is found, so a failed lookup is never cached.
    """
    t = yf.Ticker(symbol)

    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    def fi(key):
        try:
            return _num(t.fast_info[key])
        except Exception:
            return np.nan

    last_price = _first_valid(fi("last_price"), info.get("currentPrice"), info.get("regularMarketPrice"))
    shares = _first_valid(info.get("sharesOutstanding"), fi("shares"))
    market_cap = _first_valid(info.get("marketCap"), fi("market_cap"),
                              last_price * shares if pd.notna(last_price) and pd.notna(shares) else np.nan)

    # Dividend yield from trailing-12-month dividends (avoids unit ambiguity of 'dividendYield')
    div_yield_pct = np.nan
    try:
        divs = t.dividends
        if divs is not None and len(divs):
            idx = pd.DatetimeIndex(divs.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            divs = pd.Series(divs.values, index=idx)
            ttm = divs[divs.index >= (pd.Timestamp.today() - pd.Timedelta(days=365))].sum()
            if pd.notna(last_price) and last_price > 0 and ttm > 0:
                div_yield_pct = ttm / last_price * 100
    except Exception:
        pass
    if pd.isna(div_yield_pct):
        dy = _num(info.get("dividendYield"))
        if pd.notna(dy):
            # Newer yfinance returns percent already (e.g. 3.2); older returned a fraction (0.032).
            div_yield_pct = dy if dy >= 0.2 else dy * 100

    # Debt and cash from balance sheet if missing in info
    total_debt = _num(info.get("totalDebt"))
    total_cash = _num(info.get("totalCash"))
    if pd.isna(total_debt) or pd.isna(total_cash):
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                if pd.isna(total_debt):
                    total_debt = _first_valid(latest.get("Total Debt"))
                if pd.isna(total_cash):
                    total_cash = _first_valid(
                        latest.get("Cash Cash Equivalents And Short Term Investments"),
                        latest.get("Cash And Cash Equivalents"))
        except Exception:
            pass

    snap = {
        "name": info.get("longName") or info.get("shortName") or symbol,
        "short_name": info.get("shortName") or symbol,
        "last_price": last_price,
        "market_cap": market_cap,
        "shares": shares,
        "beta": _num(info.get("beta")),
        "dividend_yield_pct": div_yield_pct,
        "total_debt": total_debt,
        "total_cash": total_cash,
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "price_to_sales": _num(info.get("priceToSalesTrailing12Months")),
    }
    if pd.isna(snap["market_cap"]) and pd.isna(snap["last_price"]):
        raise RuntimeError(f"Yahoo returned no usable data for {symbol} (possible rate limit)")
    return snap


def yahoo_snapshot(symbol):
    """Returns (snapshot_dict, error_message_or_None)."""
    try:
        return _yahoo_snapshot_cached(symbol), None
    except Exception as e:
        return {}, f"{symbol}: {e}"


def annualize_beta(stock, benchmark="^GSPC"):
    """5-year regression beta of daily returns vs the S&P 500."""
    try:
        a = yahoo_history(stock, "5y")["Close"].pct_change().dropna()
        b = yahoo_history(benchmark, "5y")["Close"].pct_change().dropna()
    except Exception:
        return np.nan
    for s in (a, b):
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
    joined = pd.concat([a.rename("stock"), b.rename("market")], axis=1).dropna()
    if len(joined) < 30:
        return np.nan
    var = joined["market"].var()
    return float(joined["stock"].cov(joined["market"]) / var) if var else np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def _fundamentals_fallback(symbol, price, market_cap):
    """Compute P/E and P/S from quarterly statements when Yahoo 'info' is empty.
    Raises if nothing could be computed, so a failed lookup is never cached."""
    t = yf.Ticker(symbol)
    out = {"trailing_pe": np.nan, "price_to_sales": np.nan, "forward_pe": np.nan}

    def ttm(df, names):
        try:
            if df is None or df.empty:
                return np.nan
            for n in names:
                if n in df.index:
                    vals = pd.to_numeric(df.loc[n].iloc[:4], errors="coerce").dropna()
                    if len(vals) == 4:
                        return float(vals.sum())
        except Exception:
            pass
        return np.nan

    try:
        q = t.quarterly_income_stmt
    except Exception:
        q = None

    eps = ttm(q, ["Diluted EPS", "Basic EPS"])
    if pd.notna(price) and pd.notna(eps) and eps > 0:
        out["trailing_pe"] = price / eps

    rev = ttm(q, ["Total Revenue", "Operating Revenue"])
    if pd.notna(market_cap) and pd.notna(rev) and rev > 0:
        out["price_to_sales"] = market_cap / rev

    # Forward EPS: try several Yahoo sources, in order
    def _from_table(getter, col):
        try:
            tbl = getter()
            if tbl is not None and "+1y" in tbl.index:
                v = pd.to_numeric(tbl.loc["+1y", col], errors="coerce")
                if pd.notna(v) and v > 0:
                    return float(v)
        except Exception:
            pass
        return np.nan

    feps = _from_table(lambda: t.earnings_estimate, "avg")
    if pd.isna(feps):
        feps = _from_table(lambda: t.eps_trend, "current")
    if pd.isna(feps):
        try:
            v = pd.to_numeric(t.get_info().get("forwardEps"), errors="coerce")
            feps = float(v) if pd.notna(v) and v > 0 else np.nan
        except Exception:
            pass

    if pd.notna(price) and pd.notna(feps) and feps > 0:
        out["forward_pe"] = price / feps

    if all(pd.isna(v) for v in out.values()):
        raise RuntimeError(f"No fundamentals found for {symbol}")
    return out


def fundamentals_fallback(symbol, price, market_cap):
    try:
        return _fundamentals_fallback(symbol, price, market_cap)
    except Exception:
        return {"trailing_pe": np.nan, "price_to_sales": np.nan, "forward_pe": np.nan}


# ----------------------------- FRED -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fred_series(series_id, api_key):
    if not api_key:
        return pd.DataFrame()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).set_index("date")[["value"]]


# ----------------------------- SEC EDGAR -----------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def sec_companyfacts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def sec_fact_series(companyfacts, tag_candidates, unit_preference=None):
    """Extract annual SEC XBRL observations for a tag; prefer 10-K FY observations."""
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    tag = next((t for t in tag_candidates if t in facts), None)
    if not tag:
        return pd.Series(dtype=float, name=(tag_candidates[0] if tag_candidates else ""))
    units = facts[tag].get("units", {})
    unit = unit_preference if unit_preference in units else (
        "USD" if "USD" in units else next(iter(units), None)
    )
    if unit is None:
        return pd.Series(dtype=float, name=tag)
    rows = []
    for x in units[unit]:
        if x.get("form") != "10-K" or x.get("fp") != "FY" or not x.get("end"):
            continue
        if "start" in x:  # duration facts: keep ~annual periods only
            days = (pd.to_datetime(x["end"]) - pd.to_datetime(x["start"])).days
            if not 300 <= days <= 400:
                continue
        rows.append({"date": pd.to_datetime(x["end"]), "value": float(x["val"]),
                     "filed": x.get("filed", ""), "fy": x.get("fy")})
    if not rows:
        return pd.Series(dtype=float, name=tag)
    df = pd.DataFrame(rows).sort_values(["date", "filed"]).drop_duplicates("date", keep="last")
    s = df.set_index("date")["value"].sort_index()
    s.name = tag
    return s


def sec_fact_series_multi(companyfacts, tag_candidates):
    """Combine ALL candidate tags. Earlier tags take priority; later tags fill gaps."""
    combined = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    for tag in tag_candidates:
        s = sec_fact_series(companyfacts, [tag])
        if not s.empty:
            combined = combined.combine_first(s)
    return combined.sort_index()


def clean_sec(df, core=("Revenue", "Net income")):
    """Drop empty years/columns. Never fabricates values."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all")                  # fully empty years
    df = df.dropna(axis=1, how="all")          # metrics with no data at all
    core = [c for c in core if c in df.columns]
    if core:
        strict = df.dropna(subset=core, how="all")   # keep a year if ANY core metric exists
        if not strict.empty:
            df = strict
    return df


def get_sec_financials():
    cf = sec_companyfacts(SEC_CIK)
    definitions = {
        "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                    "SalesRevenueNet", "SalesRevenueGoodsNet"],
        "Operating income": ["OperatingIncomeLoss"],
        "Net income": ["NetIncomeLoss", "ProfitLoss"],
        "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities",
                                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "Capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets"],
        "D&A": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
        "R&D expense": ["ResearchAndDevelopmentExpense",
                        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
        "Total assets": ["Assets"],
        "Long-term debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
        "Cash": ["CashAndCashEquivalentsAtCarryingValue"],
        # Helper rows used only to derive an EBIT proxy; dropped below
        "Pretax income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                          "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
        "Interest expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    }
    out = {label: sec_fact_series_multi(cf, tags) for label, tags in definitions.items()}
    df = pd.concat(out, axis=1).sort_index()

    # Merck doesn't report an operating income line, so use an EBIT proxy (pre-tax income + interest).
    ebit_proxy = df["Pretax income"] + df["Interest expense"].fillna(0)
    df["Operating income"] = df["Operating income"].fillna(ebit_proxy)
    df = df.drop(columns=["Pretax income", "Interest expense"])

    return clean_sec(df)


def sec_quarterly_series(companyfacts, tag_candidates):
    """
    Single-quarter values for an income-statement tag.
    Q1-Q3 come from ~90-day facts in 10-Qs; Q4 = full-year (10-K) minus the 9-month year-to-date fact.
    Earlier tags in tag_candidates take priority.
    """
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    frames = []
    for prio, tag in enumerate(tag_candidates):
        units = facts.get(tag, {}).get("units", {})
        if "USD" not in units:
            continue
        rows = []
        for x in units["USD"]:
            if x.get("form") not in ("10-K", "10-Q") or "start" not in x or not x.get("end"):
                continue
            start, end = pd.to_datetime(x["start"]), pd.to_datetime(x["end"])
            rows.append({"start": start, "end": end, "days": (end - start).days,
                         "val": float(x["val"]), "filed": x.get("filed", ""), "prio": prio})
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.Series(dtype=float)

    df = pd.concat(frames, ignore_index=True)
    df = (df.sort_values(["start", "end", "prio", "filed"], ascending=[True, True, True, False])
            .drop_duplicates(["start", "end"], keep="first"))

    quarters = df[df["days"].between(80, 100)].set_index("end")["val"]
    quarters = quarters[~quarters.index.duplicated(keep="first")]

    nine_month = df[df["days"].between(260, 290)]
    q4 = {}
    for _, fy in df[df["days"].between(350, 380)].iterrows():
        match = nine_month[nine_month["start"] == fy["start"]]
        if not match.empty:
            q4[fy["end"]] = fy["val"] - match.iloc[0]["val"]

    s = pd.concat([quarters, pd.Series(q4, dtype=float)]).sort_index()
    return s[~s.index.duplicated(keep="first")]


def get_sec_quarterly_financials():
    cf = sec_companyfacts(SEC_CIK)
    definitions = {
        "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                    "SalesRevenueNet", "SalesRevenueGoodsNet"],
        "Net income": ["NetIncomeLoss", "ProfitLoss"],
        "R&D expense": ["ResearchAndDevelopmentExpense",
                        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    }
    out = {label: sec_quarterly_series(cf, tags) for label, tags in definitions.items()}
    df = pd.concat(out, axis=1).sort_index()
    return df.dropna(how="all")


# ----------------------------- ML revenue forecast -----------------------------
def build_ml_dataset(qdf):
    """Features known at quarter t -> target is revenue of quarter t+1."""
    df = qdf.sort_index().dropna(subset=["Revenue"]).copy()

    # Use only the latest unbroken run of quarters (lags are positional)
    if len(df) >= 2:
        gaps = df.index.to_series().diff().dt.days
        breaks = gaps[gaps > 120].index
        if len(breaks):
            df = df.loc[breaks[-1]:]

    rev = df["Revenue"]
    feat = pd.DataFrame(index=df.index)
    feat["Revenue"] = rev
    feat["Revenue lag 1"] = rev.shift(1)
    feat["Revenue same quarter last year"] = rev.shift(3)   # t-3 is the same quarter as the target t+1, one year earlier
    feat["QoQ growth"] = rev.pct_change(1)
    feat["YoY growth"] = rev.pct_change(4)
    if "R&D expense" in df:
        feat["R&D expense"] = df["R&D expense"]
    if "Net income" in df:
        feat["Net income"] = df["Net income"]
    target_quarter = pd.DatetimeIndex(df.index + pd.offsets.QuarterEnd(1)).quarter
    for q in (2, 3, 4):
        feat[f"Target is Q{q}"] = (target_quarter == q).astype(float)

    feat = feat.dropna(subset=["Revenue same quarter last year", "YoY growth"])
    if feat.empty:
        raise ValueError("Not enough consecutive quarterly observations to build lagged features.")

    next_dates = pd.Series(df.index, index=df.index).shift(-1).reindex(feat.index)
    y_all = rev.shift(-1).reindex(feat.index)
    has_target = y_all.notna() & next_dates.notna()

    latest_features = feat.iloc[[-1]]
    X = feat[has_target].copy()
    y = y_all[has_target].copy()
    X.index = pd.DatetimeIndex(next_dates[has_target])   # index = target quarter end
    y.index = X.index
    return X, y, latest_features


def _scaled_target(pipe):
    """Revenue is ~1e10 dollars; standardizing the target lets ElasticNet's penalty and the MLP behave."""
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


@st.cache_data(ttl=86400, show_spinner=False)
def run_ml_forecast(quarterly_df):
    X, y, latest_features = build_ml_dataset(quarterly_df)

    if len(X) < 20:
        raise ValueError(
            f"Only {len(X)} usable quarterly observations are available. "
            "At least 20 are required for this exploratory model comparison; "
            "fewer observations may produce unstable results."
        )

    # Keep the latest 20% (at least four quarters) out of training.
    test_size = max(4, int(np.ceil(len(X) * 0.20)))
    split = len(X) - test_size
    if split < 12:
        raise ValueError("Not enough training observations after the chronological split.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    def make_models():
        return {
            "Elastic Net": _scaled_target(Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
            ])),
            "Random Forest": Pipeline([  # trees do not need scaling
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, max_depth=3,
                                                min_samples_leaf=2, random_state=42)),
            ]),
            "Neural Network": _scaled_target(Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(hidden_layer_sizes=(4,), activation="relu", alpha=1.0,
                                       learning_rate_init=0.005, max_iter=5000,
                                       early_stopping=False, random_state=42)),
            ])),
        }

    models = make_models()
    metric_rows, prediction_rows = [], []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
        mae = float(mean_absolute_error(y_test, predicted))
        r_squared = float(r2_score(y_test, predicted)) if len(y_test) >= 2 else np.nan

        metric_rows.append({
            "Model": name,
            "RMSE ($bn)": rmse / 1e9,
            "MAE ($bn)": mae / 1e9,
            "R-squared": r_squared,
        })
        for period_end, actual, estimate in zip(y_test.index, y_test, predicted):
            prediction_rows.append({
                "Model": name,
                "Target quarter end": period_end.date(),
                "Actual revenue ($bn)": actual / 1e9,
                "Predicted revenue ($bn)": estimate / 1e9,
            })

    # Refit fresh models on all labeled observations to estimate the next quarter.
    next_quarter = {}
    for name, model in make_models().items():
        model.fit(X, y)
        next_quarter[name] = float(model.predict(latest_features)[0]) / 1e9

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE ($bn)")
    predictions_df = pd.DataFrame(prediction_rows)
    next_quarter_df = pd.DataFrame({
        "Model": list(next_quarter.keys()),
        "Next-quarter revenue forecast ($bn)": list(next_quarter.values()),
    })
    return metrics_df, predictions_df, next_quarter_df


# ----------------------------- Scenario model -----------------------------
def build_patent_scenarios(latest_total_revenue, latest_keytruda, years, non_key_growth,
                           pre_growth, post_erosion, cliff_year, start_year):
    rows = []
    for scenario, erosion_rate in [
        ("Bull", max(0, post_erosion * 0.6)),
        ("Base", post_erosion),
        ("Bear", min(0.85, post_erosion * 1.5)),
    ]:
        kt = latest_keytruda
        other = max(0.0, latest_total_revenue - latest_keytruda)
        for i in range(1, years + 1):
            yr = start_year + i
            other *= (1 + non_key_growth)
            if yr < cliff_year:
                kt *= (1 + pre_growth)
            else:
                kt *= (1 - erosion_rate)
            rows.append({"Scenario": scenario, "Year": yr, "Keytruda": kt,
                         "Other revenue": other, "Total revenue": kt + other,
                         "Keytruda erosion": erosion_rate})
    return pd.DataFrame(rows)


# ----------------------------- Data loading (each source isolated) -----------------------------
tabs = st.tabs(["Overview", "Market data", "Financial statements", "Patent cliff",
                "DCF valuation", "Peer comparison", "Methodology", "ML Forecast"])

load_errors = []

with st.spinner("Loading MRK market and filing data..."):
    try:
        prices = yahoo_history(TICKER, f"{PRICE_YEARS}y")
    except Exception as e:
        prices = pd.DataFrame()
        load_errors.append(f"Yahoo price history: {e}")

    snap, snap_err = yahoo_snapshot(TICKER)
    if snap_err:
        load_errors.append(f"Yahoo market snapshot: {snap_err}")

    try:
        sec_df = get_sec_financials()
    except Exception as e:
        sec_df = pd.DataFrame()
        load_errors.append(f"SEC annual financials: {e}")

    try:
        sec_q_df = get_sec_quarterly_financials()
    except Exception as e:
        sec_q_df = pd.DataFrame()
        load_errors.append(f"SEC quarterly financials for ML: {e}")

if load_errors:
    st.error("Some live data could not be loaded:\n\n" + "\n".join(f"- {m}" for m in load_errors))
    st.info("Try 'Clear cached data / retry' in the sidebar, run `pip install -U yfinance`, "
            "and note Yahoo may rate-limit requests. SEC requests require a descriptive User-Agent.")

# Normalize price data
if not prices.empty:
    prices = prices.copy()
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    prices.index = idx
    prices.index.name = "Date"
    prices = prices.dropna(subset=["Close"])
    prices["Daily return"] = prices["Close"].pct_change()
    prices["Cumulative return"] = (1 + prices["Daily return"].fillna(0)).cumprod() - 1

# Derived values with fallbacks
current_price = safe_latest(prices["Close"]) if not prices.empty else snap.get("last_price", np.nan)
market_cap = snap.get("market_cap", np.nan)
if pd.isna(market_cap) and pd.notna(current_price) and pd.notna(snap.get("shares", np.nan)):
    market_cap = current_price * snap["shares"]
beta_regression = annualize_beta(TICKER)
beta_display = snap.get("beta", np.nan)
beta_label = "Beta (Yahoo)"
if pd.isna(beta_display) and pd.notna(beta_regression):
    beta_display, beta_label = beta_regression, "Beta (5y vs S&P 500)"
div_yield_pct = snap.get("dividend_yield_pct", np.nan)

# Latest SEC fiscal year with revenue (used by the patent-cliff tab)
rev_row = latest_row(sec_df, ["Revenue"])
latest_revenue_bn = rev_row["Revenue"] / 1e9 if rev_row is not None else np.nan

# ----------------------------- Overview -----------------------------
with tabs[0]:
    st.subheader("Company snapshot")
    st.write(f"**{snap.get('name', 'Merck & Co., Inc.')}** · NYSE: MRK · Pharmaceutical industry")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest adjusted price", f"${current_price:,.2f}" if pd.notna(current_price) else "Unavailable")
    c2.metric("Market capitalization", fmt_bn(market_cap / 1e9) if pd.notna(market_cap) else "Unavailable")
    c3.metric(beta_label, f"{beta_display:.2f}" if pd.notna(beta_display) else "Unavailable")
    c4.metric("Dividend yield (TTM)", f"{div_yield_pct:.2f}%" if pd.notna(div_yield_pct) else "Unavailable")
    st.markdown("### Research question")
    st.info("How could the anticipated loss of Keytruda exclusivity affect Merck's future revenue, free cash flow, and intrinsic value?")
    st.markdown(f"### {PRICE_YEARS}-year share-price history")
    if not prices.empty:
        fig = px.line(prices, x=prices.index, y="Close", title="MRK adjusted closing price")
        fig.update_layout(xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(fig)
    st.caption("Market values are current provider snapshots; historical financial statements are based on SEC XBRL filings where available.")

# ----------------------------- Market data -----------------------------
with tabs[1]:
    st.subheader("Market performance and risk")
    if not prices.empty:
        c1, c2, c3 = st.columns(3)
        returns = prices["Daily return"].dropna()
        c1.metric("Annualized volatility", f"{returns.std() * np.sqrt(252) * 100:.1f}%")
        c2.metric(f"{PRICE_YEARS}-year cumulative return", f"{prices['Cumulative return'].iloc[-1] * 100:.1f}%")
        c3.metric("Maximum drawdown", f"{((prices['Close'] / prices['Close'].cummax()) - 1).min() * 100:.1f}%")
        st.plotly_chart(px.line(prices, x=prices.index, y="Cumulative return",
                                title="MRK cumulative total-return (dividend-adjusted)"))
        st.plotly_chart(px.bar(prices.tail(90).reset_index(), x="Date", y="Volume",
                               title="Recent trading volume"))
        st.download_button("Download MRK price history CSV", prices.to_csv().encode(), "mrk_prices.csv", "text/csv")
    else:
        st.warning("No Yahoo Finance price history is available.")
    st.markdown("### FRED macroeconomic indicators")
    series_choice = st.selectbox("FRED series", list(FRED_DEFAULT_SERIES.keys()))
    sid = FRED_DEFAULT_SERIES[series_choice]
    if fred_key:
        try:
            macro = fred_series(sid, fred_key)
            if not macro.empty:
                st.line_chart(macro)
                st.download_button("Download FRED data CSV", macro.to_csv().encode(), f"{sid}.csv", "text/csv")
            else:
                st.warning("No observations returned for this FRED series.")
        except Exception as e:
            st.error(f"FRED request failed: {e}")
    else:
        st.info(f"Enter a FRED API key in the sidebar to retrieve {sid}. For the DCF, you can use the manual risk-free rate assumption.")

# ----------------------------- Financial statements -----------------------------
with tabs[2]:
    st.subheader("SEC-reported annual financial data")
    if not sec_df.empty:
        display = sec_df.copy() / 1e9

        chosen = [c for c in ["Revenue", "Operating income", "Net income", "Operating cash flow", "R&D expense"]
                  if c in display and display[c].notna().any()]

        table = display.copy()
        table.index = table.index.year
        table.index.name = "Fiscal year"
        show_df(table)
        st.caption("USD billions. 'Operating income' is an EBIT proxy (pre-tax income + interest expense) where Merck "
                   "does not report it directly. '—' means no value was reported under the tags used. "
                   "Verify all figures against the original 10-K.")

        if chosen:
            chart_df = display[chosen].reset_index(names="Fiscal year end").melt(
                "Fiscal year end", var_name="Metric", value_name="USD billions").dropna()
            st.plotly_chart(px.line(chart_df, x="Fiscal year end", y="USD billions", color="Metric",
                                    markers=True, title="Historical financial performance"))
        st.download_button("Download SEC financial data CSV", sec_df.to_csv().encode(), "mrk_sec_financials.csv", "text/csv")
    else:
        st.warning("SEC companyfacts data could not be loaded. Check the SEC request or use the company's filings directly.")

    with st.expander("Yahoo Finance statement cross-check"):
        try:
            statements = yahoo_statements(TICKER)
            for key, label in [("income", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")]:
                st.markdown(f"**{label}**")
                show_df(statements[key].dropna(how="all"), "{:,.0f}")
        except Exception as e:
            st.warning(f"Yahoo statements unavailable: {e}")

# ----------------------------- Patent cliff -----------------------------
with tabs[3]:
    st.subheader("Keytruda patent-cliff scenario model")
    st.markdown(
        "This module separates Keytruda from the rest of Merck's revenue, then applies explicit scenario assumptions. "
        "The starting Keytruda revenue is a user-editable input: verify it against Merck's latest annual report."
    )
    pc_revenue_bn = latest_revenue_bn
    sec_revenue_ok = bool(np.isfinite(pc_revenue_bn) and pc_revenue_bn > 0)
    if not sec_revenue_ok:
        pc_revenue_bn = 60.0
        st.warning("SEC revenue could not be read; using an illustrative total-revenue starting value of $60B. Replace it with the latest reported value.")
    st.metric("Latest SEC annual revenue" if sec_revenue_ok else "Starting revenue (illustrative fallback)",
              fmt_bn(pc_revenue_bn))
    if keytruda_revenue >= pc_revenue_bn:
        st.warning("Keytruda revenue input is greater than or equal to total company revenue. Check the reporting year and units.")
    scenarios = build_patent_scenarios(
        pc_revenue_bn, keytruda_revenue, forecast_years,
        non_keytruda_growth / 100, pre_cliff_growth / 100,
        erosion / 100, int(cliff_start_year), as_of.year
    )
    fig = px.line(scenarios, x="Year", y="Total revenue", color="Scenario", markers=True,
                  title="Projected total revenue by patent-cliff scenario")
    fig.update_layout(yaxis_title="Revenue ($bn)")
    st.plotly_chart(fig)
    fig2 = px.line(scenarios, x="Year", y="Keytruda", color="Scenario", markers=True,
                   title="Illustrative Keytruda revenue erosion")
    fig2.update_layout(yaxis_title="Revenue ($bn)")
    st.plotly_chart(fig2)
    st.dataframe(scenarios.pivot(index="Year", columns="Scenario", values="Total revenue")
                 .style.format("${:,.1f}B", na_rep="—"))
    st.download_button("Download patent-cliff scenarios CSV", scenarios.to_csv(index=False).encode(),
                       "mrk_patent_scenarios.csv", "text/csv")
    st.caption("This is a simplified scenario model, not a company forecast. It does not model product-level launches, geographic patent differences, litigation outcomes, or volume/price dynamics.")

# ----------------------------- DCF valuation -----------------------------
with tabs[4]:
    st.subheader("Discounted cash flow valuation")
    st.markdown("The DCF is an illustrative FCFF model. Review and replace assumptions with your own researched estimates.")

    # Revenue and EBIT must come from the SAME fiscal year
    row = latest_row(sec_df, ["Revenue", "Operating income"])
    if row is not None:
        rev_bn = row["Revenue"] / 1e9
        op_income_bn = row["Operating income"] / 1e9
    else:
        rev_bn = latest_revenue_bn
        op_income_bn = np.nan
    if not np.isfinite(rev_bn) or rev_bn <= 0:
        rev_bn = 60.0
    op_margin_default = (max(0.05, min(0.50, op_income_bn / rev_bn))
                         if np.isfinite(op_income_bn) else 0.25)

    c1, c2, c3 = st.columns(3)
    with c1:
        starting_revenue = st.number_input("Starting revenue ($bn)", min_value=1.0, value=float(round(rev_bn, 2)), step=1.0)
        starting_ebit_margin = st.number_input("Starting EBIT margin (%)", min_value=-10.0, max_value=60.0,
                                               value=float(round(op_margin_default * 100, 1)), step=0.5)
    with c2:
        da_pct = st.number_input("D&A / revenue (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        capex_pct = st.number_input("CapEx / revenue (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
    with c3:
        nwc_pct = st.number_input("Change in NWC / revenue (%)", min_value=-10.0, max_value=20.0, value=1.0, step=0.5)
        dcf_scenario = st.selectbox("Patent-cliff scenario linked to DCF", ["Base", "Bull", "Bear"])

    dcf_scenarios = build_patent_scenarios(
        starting_revenue, keytruda_revenue, forecast_years,
        non_keytruda_growth / 100, pre_cliff_growth / 100,
        erosion / 100, int(cliff_start_year), as_of.year
    )
    selected_revenue_path = (
        dcf_scenarios[dcf_scenarios["Scenario"] == dcf_scenario]
        .sort_values("Year")["Total revenue"].tolist()
    )
    if len(selected_revenue_path) != forecast_years:
        selected_revenue_path = [starting_revenue * (1 + 0.03) ** i for i in range(1, forecast_years + 1)]
        st.warning("Could not construct the selected patent-cliff revenue path; using a 3% illustrative fallback.")

    st.markdown("#### Capital structure and discount rate")
    c1, c2, c3, c4 = st.columns(4)
    beta_default = beta_display if pd.notna(beta_display) and 0 <= beta_display <= 4 else 0.80
    beta_used = c1.number_input("Levered beta", min_value=0.0, max_value=4.0,
                                value=float(round(beta_default, 2)), step=0.05)
    # These three come from the sidebar (re-creating the same widgets here caused duplicate-ID errors).
    rf = float(risk_free_manual)
    erp = float(equity_risk_premium)
    debt_cost = float(cost_of_debt)
    c2.metric("Risk-free rate", f"{rf:.1f}%")
    c3.metric("Equity risk premium", f"{erp:.1f}%")
    c4.metric("Pre-tax cost of debt", f"{debt_cost:.1f}%")
    st.caption("Risk-free rate, equity risk premium and cost of debt are set in the sidebar.")

    default_equity = float(market_cap / 1e9) if pd.notna(market_cap) and market_cap > 0 else 300.0
    td = snap.get("total_debt", np.nan)
    tc = snap.get("total_cash", np.nan)
    default_debt = float(td / 1e9) if pd.notna(td) and td > 0 else 30.0
    default_cash = float(tc / 1e9) if pd.notna(tc) and tc > 0 else 10.0
    c1, c2, c3 = st.columns(3)
    equity_value = c1.number_input("Market value of equity ($bn)", min_value=1.0, value=default_equity, step=5.0)
    debt_value = c2.number_input("Total debt ($bn)", min_value=0.0, value=default_debt, step=1.0)
    cash_value = c3.number_input("Cash & equivalents ($bn)", min_value=0.0, value=default_cash, step=1.0)
    if pd.isna(market_cap) or pd.isna(td) or pd.isna(tc):
        st.warning("Some market inputs (market cap, debt or cash) could not be fetched; placeholder defaults are used. Please verify them.")

    tax = tax_rate / 100
    cost_equity_pct = rf + beta_used * erp
    total_cap = equity_value + debt_value
    wacc = ((equity_value / total_cap) * (cost_equity_pct / 100) +
            (debt_value / total_cap) * (debt_cost / 100) * (1 - tax)) if total_cap else np.nan
    g = terminal_growth / 100

    if not np.isfinite(wacc) or wacc <= g:
        st.error("WACC must be a valid number greater than the terminal growth rate for the Gordon Growth terminal value.")
    else:
        rows = []
        for year in range(1, forecast_years + 1):
            rev = selected_revenue_path[year - 1]
            ebit = rev * starting_ebit_margin / 100
            nopat = ebit * (1 - tax)
            da = rev * da_pct / 100
            capex = rev * capex_pct / 100
            delta_nwc = rev * nwc_pct / 100
            fcff = nopat + da - capex - delta_nwc
            rows.append({"Year": as_of.year + year, "Revenue": rev, "EBIT": ebit, "NOPAT": nopat,
                         "D&A": da, "CapEx": capex, "Change in NWC": delta_nwc, "FCFF": fcff})
        forecast = pd.DataFrame(rows)
        forecast["Discount factor"] = [1 / (1 + wacc) ** i for i in range(1, forecast_years + 1)]
        forecast["PV of FCFF"] = forecast["FCFF"] * forecast["Discount factor"]
        terminal_value = forecast["FCFF"].iloc[-1] * (1 + g) / (wacc - g)
        pv_terminal = terminal_value / (1 + wacc) ** forecast_years
        ev = forecast["PV of FCFF"].sum() + pv_terminal
        eq_val = ev - (debt_value - cash_value)
        shares = snap.get("shares", np.nan)
        price_per_share = eq_val * 1e9 / shares if pd.notna(shares) and shares > 0 else np.nan

        a, b, c, d = st.columns(4)
        a.metric("WACC", f"{wacc * 100:.2f}%")
        b.metric("Enterprise value", fmt_bn(ev))
        c.metric("Equity value", fmt_bn(eq_val))
        d.metric("Implied value / share", f"${price_per_share:,.2f}" if pd.notna(price_per_share) else "Shares unavailable")
        money_cols = ["Revenue", "EBIT", "NOPAT", "D&A", "CapEx", "Change in NWC", "FCFF", "PV of FCFF"]
        fmt_map = {col: "${:,.2f}B" for col in money_cols}
        fmt_map["Discount factor"] = "{:.4f}"
        st.dataframe(forecast.style.format(fmt_map, na_rep="—"))
        st.plotly_chart(px.bar(forecast, x="Year", y="FCFF", title="Forecast free cash flow to firm"))
        st.write(f"Terminal value: **{fmt_bn(terminal_value)}** · Present value of terminal value: **{fmt_bn(pv_terminal)}**")
        st.caption("The DCF revenue forecast is linked to the selected patent-cliff scenario. EBIT margin, D&A, CapEx, and working-capital assumptions remain simplified and should be researched and refined.")
    st.caption("This simplified model is sensitive to assumptions. Verify market capitalization, debt, cash, shares outstanding, beta, and all accounting inputs as of a consistent valuation date.")

# ----------------------------- Peer comparison -----------------------------
with tabs[5]:
    st.subheader("Pharmaceutical peer comparison")
    peer_rows, peer_errors = [], []

    with st.expander("Forward EPS estimates (optional: enter next-year consensus EPS)"):
        _syms = [TICKER] + PEERS
        _cols = st.columns(len(_syms))
        manual_eps = {}
        for _c, _sym in zip(_cols, _syms):
            manual_eps[_sym] = _c.number_input(f"{_sym} EPS ($)", min_value=0.0, value=0.0,
                                               step=0.1, key=f"feps_{_sym}")

    for sym in [TICKER] + PEERS:
        s, err = yahoo_snapshot(sym)
        if err:
            peer_errors.append(err)
        fb = fundamentals_fallback(sym, s.get("last_price", np.nan), s.get("market_cap", np.nan))

        def pick(key, s=s, fb=fb):
            v = s.get(key, np.nan)
            return v if pd.notna(v) else fb[key]

        # Forward P/E: user-entered EPS overrides the automatic value
        fwd_pe = pick("forward_pe")
        eps_in = manual_eps.get(sym, 0.0)
        px_now = s.get("last_price", np.nan)
        if eps_in > 0 and pd.notna(px_now):
            fwd_pe = px_now / eps_in

        beta_val = s.get("beta", np.nan)
        if pd.isna(beta_val):
            beta_val = annualize_beta(sym)   # 5y regression vs S&P 500

        mcap = s.get("market_cap", np.nan)
        peer_rows.append({
            "Ticker": sym,
            "Company": s.get("short_name") if s.get("short_name") not in (None, sym) else PEER_NAMES.get(sym, sym),
            "Market cap ($bn)": mcap / 1e9 if pd.notna(mcap) else np.nan,
            "Trailing P/E": pick("trailing_pe"),
            "Forward P/E": fwd_pe,
            "Price / Sales": pick("price_to_sales"),
            "Dividend yield (%)": s.get("dividend_yield_pct", np.nan),
            "Beta": beta_val,
        })

    peer_df = pd.DataFrame(peer_rows)
    num_cols = [c for c in peer_df.columns if c not in ("Ticker", "Company")]
    peer_df[num_cols] = peer_df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Turn every cell into text so no "None"/"nan" can ever appear
    formats = {"Market cap ($bn)": "${:,.1f}", "Trailing P/E": "{:.1f}x", "Forward P/E": "{:.1f}x",
               "Price / Sales": "{:.1f}x", "Dividend yield (%)": "{:.2f}%", "Beta": "{:.2f}"}
    shown = peer_df.copy()
    for col, f in formats.items():
        shown[col] = peer_df[col].apply(lambda v, f=f: "—" if pd.isna(v) else f.format(v))
    st.dataframe(shown, hide_index=True)

    if peer_errors:
        st.warning("Some peer data could not be loaded: " + "; ".join(peer_errors))
    st.caption("Peer data are provider-supplied snapshots and may be missing or calculated on different bases. "
               "Forward P/E uses automatic estimates where available, or the EPS you entered above. "
               "'—' means the value could not be retrieved or is not meaningful (e.g. negative earnings). "
               "Confirm comparability before drawing conclusions.")
    st.download_button("Download peer comparison CSV", peer_df.to_csv(index=False).encode(),
                       "mrk_peer_comparison.csv", "text/csv")

# ----------------------------- Methodology -----------------------------
with tabs[6]:
    st.subheader("Methodology, assumptions & limitations")
    st.markdown("""
**Data hierarchy**
1. SEC EDGAR 10-K/10-Q filings for reported financial statements (all candidate XBRL tags are combined; earlier tags take priority and later tags fill gaps).
2. Merck annual reports and earnings releases for product-level sales and management guidance.
3. Yahoo Finance for market prices, dividends, market capitalization, and peer snapshots (with fallbacks: fast_info, balance sheet, dividend history, regression beta).
4. FRED for Treasury yields, rates, and macroeconomic series.

**Core formulas**
- Daily return: `P(t) / P(t-1) - 1`
- Dividend yield: `trailing-12-month dividends / latest price`
- CAPM cost of equity: `Risk-free rate + beta × equity risk premium`
- WACC: `E/(D+E) × cost of equity + D/(D+E) × cost of debt × (1-tax rate)`
- FCFF: `EBIT × (1-tax) + D&A - CapEx - change in NWC`
- Terminal value: `FCFF(n+1) / (WACC - terminal growth)`
- Equity value: `Enterprise value - debt + cash`
- EBIT proxy (when operating income is not reported): `Pre-tax income + interest expense`

**Machine-learning forecast**
- Target: next quarter's revenue. Features: current and lagged revenue, QoQ and YoY growth, R&D expense, net income, and next-quarter seasonality dummies.
- Quarterly figures come from SEC filings; Q4 is the full-year figure minus the nine-month year-to-date figure.
- Chronological split: the latest 20% of quarters (at least four) are held out. The target is standardized for Elastic Net and the neural network.

**Data handling**
- Missing values are never filled with zeros, forward-fills or interpolation in the financial tables. Gaps display as "—". (The ML pipeline imputes missing feature values with the training median.)
- Revenue and EBIT margin in the DCF come from the same fiscal year.

**Important limitations**
- Patent expiration and biosimilar launch dates are legal and commercial uncertainties; verify the specific patents, geographies, and litigation status from primary sources.
- The patent-cliff model is a simplified revenue bridge, not a pharmaceutical product-level forecast.
- The DCF is a teaching model. It should be improved with explicit product/segment forecasts, margin drivers, tax assumptions, and a consistent balance-sheet bridge.
- The ML comparison uses a small sample and a short test window; treat it as exploratory, not predictive evidence.
- Yahoo Finance data are not guaranteed to be complete or stable. SEC XBRL tags may require manual reconciliation to reported statements.
- Historical stock returns are not forecasts of future returns.
- Do not treat the output as a buy/sell recommendation.
""")
    st.markdown("**Suggested sources:** SEC EDGAR company filings (CIK 0000310158), Merck investor relations, Yahoo Finance, and FRED.")
    st.code("pip install -U streamlit yfinance pandas numpy plotly requests scikit-learn\nstreamlit run app.py",
            language="bash")

# ----------------------------- ML Forecast -----------------------------
with tabs[7]:
    st.subheader("Machine-learning revenue forecast (exploratory)")
    st.markdown("Compares Elastic Net, Random Forest and a small neural network at predicting **next quarter's revenue** "
                "from SEC quarterly filings, using a chronological hold-out of the most recent quarters.")
    if sec_q_df.empty:
        st.warning("SEC quarterly data could not be loaded, so the ML comparison is unavailable.")
    else:
        try:
            with st.spinner("Training models..."):
                metrics_df, predictions_df, next_quarter_df = run_ml_forecast(sec_q_df)
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"ML forecast failed: {e}")
        else:
            st.markdown("#### Hold-out accuracy (lower RMSE/MAE is better)")
            st.dataframe(metrics_df.style.format(
                {"RMSE ($bn)": "{:.2f}", "MAE ($bn)": "{:.2f}", "R-squared": "{:.2f}"}, na_rep="—"),
                hide_index=True)

            actual_df = (predictions_df.drop_duplicates("Target quarter end")
                         [["Target quarter end", "Actual revenue ($bn)"]]
                         .rename(columns={"Actual revenue ($bn)": "Revenue ($bn)"}))
            actual_df["Model"] = "Actual"
            pred_df = predictions_df.rename(columns={"Predicted revenue ($bn)": "Revenue ($bn)"})[
                ["Target quarter end", "Model", "Revenue ($bn)"]]
            chart_df = pd.concat([actual_df[["Target quarter end", "Model", "Revenue ($bn)"]], pred_df])
            st.plotly_chart(px.line(chart_df, x="Target quarter end", y="Revenue ($bn)", color="Model",
                                    markers=True, title="Hold-out quarters: actual vs predicted revenue"))

            st.markdown("#### Next-quarter revenue forecast")
            st.dataframe(next_quarter_df.style.format({"Next-quarter revenue forecast ($bn)": "${:,.2f}B"}),
                         hide_index=True)
            st.download_button("Download hold-out predictions CSV",
                               predictions_df.to_csv(index=False).encode(), "mrk_ml_predictions.csv", "text/csv")
            st.caption("Small sample, short test window, no hyperparameter tuning. Tree models cannot extrapolate beyond "
                       "the range of revenue seen in training. Not investment advice.")

st.markdown("---")
st.caption(f"Built for educational use · Valuation date selected: {as_of.isoformat()} · Data retrieval is subject to provider availability and API terms.")
