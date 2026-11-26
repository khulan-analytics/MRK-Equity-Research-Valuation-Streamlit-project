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

    models = {
        "Elastic Net": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42
            ))
        ]),
        "Neural Network": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(4,),
                activation="relu",
                alpha=1.0,
                learning_rate_init=0.005,
                max_iter=5000,
                early_stopping=False,
                random_state=42
            ))
        ])
    }

    metric_rows = []
    prediction_rows = []

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
            "R-squared": r_squared
        })

        for period_end, actual, estimate in zip(y_test.index, y_test, predicted):
            prediction_rows.append({
                "Model": name,
                "Target quarter end": period_end.date(),
                "Actual revenue ($bn)": actual / 1e9,
                "Predicted revenue ($bn)": estimate / 1e9
            })

    # Refit on all labeled observations to estimate the next quarter.
    next_quarter = {}
    for name, model in models.items():
        model.fit(X, y)
        estimate = float(model.predict(latest_features)[0])
        next_quarter[name] = estimate / 1e9

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE ($bn)")
    predictions_df = pd.DataFrame(prediction_rows)
    next_quarter_df = pd.DataFrame({
        "Model": list(next_quarter.keys()),
        "Next-quarter revenue forecast ($bn)": list(next_quarter.values())
    })

    return metrics_df, predictions_df, next_quarter_df


# ----------------------------- Scenario model -----------------------------
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
                "DCF valuation", "Peer comparison", "Methodology", "ML Forecast"])

load_errors = []

with st.spinner("Loading MRK market and filing data..."):
    try:
        prices = yahoo_history(TICKER, "10y")
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
