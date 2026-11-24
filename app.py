from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

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
TICKER = "MRK"

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


# ----------------------------- Helpers -----------------------------
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

    # 1) Full info (scraped endpoint; often rate-limited or empty)
    try:
        info = t.get_info() or {}
    except Exception:
        info = {}

    # 2) fast_info (lighter endpoint)
    def fi(key):
        try:
            return _num(t.fast_info[key])
        except Exception:
            return np.nan

    last_price = _first_valid(fi("last_price"), info.get("currentPrice"), info.get("regularMarketPrice"))
    shares = _first_valid(info.get("sharesOutstanding"), fi("shares"))
    market_cap = _first_valid(info.get("marketCap"), fi("market_cap"),
                              last_price * shares if pd.notna(last_price) and pd.notna(shares) else np.nan)

    # 3) Dividend yield from trailing-12-month dividends (avoids unit ambiguity of 'dividendYield')
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

    # 4) Debt and cash from balance sheet if missing in info
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


@st.cache_data(ttl=86400, show_spinner=False)
def sec_companyfacts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


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


def get_sec_financials():
    cf = sec_companyfacts(SEC_CIK)
    definitions = {
        "Revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "Operating income": ["OperatingIncomeLoss"],
        "Net income": ["NetIncomeLoss"],
        "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
        "Capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
        "D&A": ["DepreciationDepletionAndAmortizationPropertyPlantAndEquipment", "DepreciationDepletionAndAmortization"],
        "R&D expense": ["ResearchAndDevelopmentExpense"],
        "Total assets": ["Assets"],
        # Noncurrent debt first (the old order picked the current portion only)
        "Long-term debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtCurrent"],
        "Cash": ["CashAndCashEquivalentsAtCarryingValue"],
    }
    out = {label: sec_fact_series(cf, tags) for label, tags in definitions.items()}
    return pd.concat(out, axis=1).sort_index()


def fmt_bn(x):
    return "—" if pd.isna(x) else f"${x:,.2f}B"


def safe_latest(series, default=np.nan):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[-1]) if len(s) else default


def annualize_beta(stock, benchmark="^GSPC"):
    try:
        a = yahoo_history(stock, "5y")["Close"].pct_change().dropna()
        b = yahoo_history(benchmark, "5y")["Close"].pct_change().dropna()
    except Exception:
        return np.nan
    a.index = pd.DatetimeIndex(a.index).tz_localize(None) if pd.DatetimeIndex(a.index).tz is not None else a.index
    b.index = pd.DatetimeIndex(b.index).tz_localize(None) if pd.DatetimeIndex(b.index).tz is not None else b.index
    a.index = a.index.normalize()
    b.index = b.index.normalize()
    joined = pd.concat([a.rename("stock"), b.rename("market")], axis=1).dropna()
    if len(joined) < 30:
        return np.nan
    var = joined["market"].var()
    return float(joined["stock"].cov(joined["market"]) / var) if var else np.nan


def build_patent_scenarios(latest_total_revenue, latest_keytruda, years, non_key_growth,
                           pre_growth, post_erosion, cliff_year):
    rows = []
    start_year = date.today().year
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
                "DCF valuation", "Peer comparison", "Methodology"])

load_errors = []

with st.spinner("Loading MRK market and filing data..."):
    try:
        prices = yahoo_history(TICKER, "5y")
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
        load_errors.append(f"SEC EDGAR: {e}")

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
    st.markdown("### Five-year share-price history")
    if not prices.empty:
        fig = px.line(prices, x=prices.index, y="Close", title="MRK adjusted closing price")
        fig.update_layout(xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Market values are current provider snapshots; historical financial statements are based on SEC XBRL filings where available.")

# ----------------------------- Market data -----------------------------
with tabs[1]:
    st.subheader("Market performance and risk")
    if not prices.empty:
        c1, c2, c3 = st.columns(3)
        returns = prices["Daily return"].dropna()
        c1.metric("Annualized volatility", f"{returns.std()*np.sqrt(252)*100:.1f}%")
        c2.metric("5-year cumulative return", f"{prices['Cumulative return'].iloc[-1]*100:.1f}%")
        c3.metric("Maximum drawdown", f"{((prices['Close']/prices['Close'].cummax())-1).min()*100:.1f}%")
        st.plotly_chart(px.line(prices, x=prices.index, y="Cumulative return",
                                title="MRK cumulative total-return (dividend-adjusted)"), use_container_width=True)
        st.plotly_chart(px.bar(prices.tail(90).reset_index(), x="Date", y="Volume",
                               title="Recent trading volume"), use_container_width=True)
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
        st.dataframe(display.style.format("{:,.2f}"), use_container_width=True)
        st.caption("USD billions. SEC companyfacts tags and available fiscal-year observations can differ by company and filing. Verify all figures against the original 10-K.")
        chosen = [c for c in ["Revenue", "Operating income", "Net income", "Operating cash flow", "R&D expense"]
                  if c in display and display[c].notna().any()]
        if chosen:
            chart_df = display[chosen].reset_index(names="Fiscal year end").melt(
                "Fiscal year end", var_name="Metric", value_name="USD billions")
            st.plotly_chart(px.line(chart_df, x="Fiscal year end", y="USD billions", color="Metric",
                                    markers=True, title="Historical financial performance"),
                            use_container_width=True)
        st.download_button("Download SEC financial data CSV", sec_df.to_csv().encode(), "mrk_sec_financials.csv", "text/csv")
    else:
        st.warning("SEC companyfacts data could not be loaded. Check the SEC request or use the company's filings directly.")
    with st.expander("Yahoo Finance statement cross-check"):
        try:
            statements = yahoo_statements(TICKER)
            for key, label in [("income", "Income statement"), ("balance", "Balance sheet"), ("cashflow", "Cash flow")]:
                st.markdown(f"**{label}**")
                st.dataframe(statements[key], use_container_width=True)
        except Exception as e:
            st.warning(f"Yahoo statements unavailable: {e}")

# ----------------------------- Patent cliff -----------------------------
with tabs[3]:
    st.subheader("Keytruda patent-cliff scenario model")
    st.markdown(
        "This module separates Keytruda from the rest of Merck's revenue, then applies explicit scenario assumptions. "
        "The starting Keytruda revenue is a user-editable input: verify it against Merck's latest annual report."
    )
    latest_revenue_bn = safe_latest(sec_df.get("Revenue", pd.Series(dtype=float))) / 1e9
    if not np.isfinite(latest_revenue_bn) or latest_revenue_bn <= 0:
        latest_revenue_bn = 60.0
        st.warning("SEC revenue could not be read; using an illustrative total-revenue starting value of $60B. Replace it with the latest reported value.")
    st.metric("Latest SEC annual revenue (if available)", fmt_bn(latest_revenue_bn))
    if keytruda_revenue >= latest_revenue_bn:
        st.warning("Keytruda revenue input is greater than or equal to total company revenue. Check the reporting year and units.")
    scenarios = build_patent_scenarios(
        latest_revenue_bn, keytruda_revenue, forecast_years,
        non_keytruda_growth / 100, pre_cliff_growth / 100,
        erosion / 100, int(cliff_start_year)
    )
    fig = px.line(scenarios, x="Year", y="Total revenue", color="Scenario", markers=True,
                  title="Projected total revenue by patent-cliff scenario")
    fig.update_layout(yaxis_title="Revenue ($bn)")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.line(scenarios, x="Year", y="Keytruda", color="Scenario", markers=True,
                   title="Illustrative Keytruda revenue erosion")
    fig2.update_layout(yaxis_title="Revenue ($bn)")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(scenarios.pivot(index="Year", columns="Scenario", values="Total revenue")
                 .style.format("${:,.1f}B"), use_container_width=True)
    st.download_button("Download patent-cliff scenarios CSV", scenarios.to_csv(index=False).encode(),
                       "mrk_patent_scenarios.csv", "text/csv")
    st.caption("This is a simplified scenario model, not a company forecast. It does not model product-level launches, geographic patent differences, litigation outcomes, or volume/price dynamics.")

# ----------------------------- DCF valuation -----------------------------
with tabs[4]:
    st.subheader("Discounted cash flow valuation")
    st.markdown("The DCF is an illustrative FCFF model. Review and replace assumptions with your own researched estimates.")
    rev_bn = safe_latest(sec_df.get("Revenue", pd.Series(dtype=float))) / 1e9
    if not np.isfinite(rev_bn) or rev_bn <= 0:
        rev_bn = 60.0
    op_income_bn = safe_latest(sec_df.get("Operating income", pd.Series(dtype=float))) / 1e9
    op_margin_default = (max(0.05, min(0.50, op_income_bn / rev_bn))
                         if np.isfinite(op_income_bn) and rev_bn else 0.25)
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
        erosion / 100, int(cliff_start_year)
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
    beta_used = c1.number_input("Levered beta", min_value=0.0, max_value=4.0, value=float(round(beta_default, 2)), step=0.05)
    rf = c2.number_input("Risk-free rate (%)", min_value=0.0, max_value=15.0, value=float(risk_free_manual), step=0.1)
    erp = c3.number_input("Equity risk premium (%)", min_value=0.0, max_value=15.0, value=float(equity_risk_premium), step=0.1)
    debt_cost = c4.number_input("Pre-tax cost of debt (%)", min_value=0.0, max_value=20.0, value=float(cost_of_debt), step=0.1)

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
            rows.append({"Year": date.today().year + year, "Revenue": rev, "EBIT": ebit, "NOPAT": nopat,
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
        a.metric("WACC", f"{wacc*100:.2f}%")
        b.metric("Enterprise value", fmt_bn(ev))
        c.metric("Equity value", fmt_bn(eq_val))
        d.metric("Implied value / share", f"${price_per_share:,.2f}" if pd.notna(price_per_share) else "Shares unavailable")
        money_cols = ["Revenue", "EBIT", "NOPAT", "D&A", "CapEx", "Change in NWC", "FCFF", "PV of FCFF"]
        st.dataframe(forecast.style.format({col: "${:,.2f}B" for col in money_cols}), use_container_width=True)
        st.plotly_chart(px.bar(forecast, x="Year", y="FCFF", title="Forecast free cash flow to firm"), use_container_width=True)
        st.write(f"Terminal value: **{fmt_bn(terminal_value)}** · Present value of terminal value: **{fmt_bn(pv_terminal)}**")
        st.caption("The DCF revenue forecast is linked to the selected patent-cliff scenario. EBIT margin, D&A, CapEx, and working-capital assumptions remain simplified and should be researched and refined.")
    st.caption("This simplified model is sensitive to assumptions. Verify market capitalization, debt, cash, shares outstanding, beta, and all accounting inputs as of a consistent valuation date.")

# ----------------------------- Peer comparison -----------------------------
with tabs[5]:
    st.subheader("Pharmaceutical peer comparison")
    peer_rows, peer_errors = [], []
    for sym in [TICKER] + PEERS:
        s, err = yahoo_snapshot(sym)
        if err:
            peer_errors.append(err)
        peer_rows.append({
            "Ticker": sym,
            "Company": s.get("short_name", sym),
            "Market cap ($bn)": s["market_cap"] / 1e9 if pd.notna(s.get("market_cap", np.nan)) else np.nan,
            "Trailing P/E": s.get("trailing_pe", np.nan),
            "Forward P/E": s.get("forward_pe", np.nan),
            "Price / Sales": s.get("price_to_sales", np.nan),
            "Dividend yield (%)": s.get("dividend_yield_pct", np.nan),
            "Beta": s.get("beta", np.nan),
        })
    peer_df = pd.DataFrame(peer_rows)
    st.dataframe(peer_df.style.format({
        "Market cap ($bn)": "${:,.1f}", "Trailing P/E": "{:.1f}x", "Forward P/E": "{:.1f}x",
        "Price / Sales": "{:.1f}x", "Dividend yield (%)": "{:.2f}%", "Beta": "{:.2f}"
    }, na_rep="—"), use_container_width=True)
    if peer_errors:
        st.warning("Some peer data could not be loaded: " + "; ".join(peer_errors))
    st.caption("Peer data are provider-supplied snapshots and may be missing or calculated on different bases. Confirm comparability before drawing conclusions.")
    st.download_button("Download peer comparison CSV", peer_df.to_csv(index=False).encode(),
                       "mrk_peer_comparison.csv", "text/csv")

# ----------------------------- Methodology -----------------------------
with tabs[6]:
    st.subheader("Methodology, assumptions & limitations")
    st.markdown("""
**Data hierarchy**
1. SEC EDGAR 10-K/10-Q filings for reported financial statements.
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

**Important limitations**
- Patent expiration and biosimilar launch dates are legal and commercial uncertainties; verify the specific patents, geographies, and litigation status from primary sources.
- The patent-cliff model is a simplified revenue bridge, not a pharmaceutical product-level forecast.
- The DCF is a teaching model. It should be improved with explicit product/segment forecasts, margin drivers, tax assumptions, and a consistent balance-sheet bridge.
- Yahoo Finance data are not guaranteed to be complete or stable. SEC XBRL tags may require manual reconciliation to reported statements.
- Historical stock returns are not forecasts of future returns.
- Do not treat the output as a buy/sell recommendation.
""")
    st.markdown("**Suggested sources:** SEC EDGAR company filings (CIK 0000310158), Merck investor relations, Yahoo Finance, and FRED.")
    st.code("pip install -U streamlit yfinance pandas numpy plotly requests\nstreamlit run app.py", language="bash")

st.markdown("---")
st.caption(f"Built for educational use · Valuation date selected: {as_of.isoformat()} · Data retrieval is subject to provider availability and API terms.")
