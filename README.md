# MRK Equity Research & Valuation Dashboard

A Streamlit educational project for Merck & Co. (NYSE: MRK), combining:
- Yahoo Finance market data
- SEC EDGAR companyfacts XBRL data
- Optional FRED macroeconomic series
- Keytruda patent-cliff scenario modeling
- A simplified FCFF discounted cash flow model
- Pharmaceutical peer comparison

## Run locally

1. Install Python 3.10+.
2. Unzip the project.
3. In a terminal, enter the project directory.
4. Create and activate a virtual environment (recommended).
5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Start Streamlit:

```bash
streamlit run app.py
```

## Important setup and data notes

- SEC requests identify the application with a User-Agent. Replace the placeholder in `app.py` (`student.project@example.com`) with a valid contact email before repeated use.
- A FRED API key is optional for the FRED tab. Without it, the DCF uses a manually entered risk-free rate.
- Yahoo Finance access and fields may change. Some peer metrics may be missing.
- SEC companyfacts contains standardized XBRL facts, not a perfectly reconciled financial model. Verify all figures against Merck's 10-K and 10-Q filings.
- The Keytruda revenue input is editable and must be updated from Merck's latest reported product sales. Do not assume the prefilled value is current.
- The patent-cliff scenarios are hypothetical assumptions, not Merck guidance or predictions.
- The DCF is deliberately simplified for a course project. Improve it by connecting the patent-cliff scenario revenue path to the forecast, explicitly forecasting margins and working capital, reconciling debt/cash/share count, and documenting every assumption.

## Suggested project workflow

1. Validate SEC financial statements and reconcile units/fiscal periods.
2. Verify Keytruda sales and patent disclosures from Merck primary filings.
3. Build product-level revenue scenarios.
4. Link those scenarios into the FCFF forecast.
5. Validate WACC inputs and valuation date.
6. Add tests for formulas and data transformations.
7. Deploy only after reviewing data-provider terms and secrets management.

## Security

Do not commit API keys or personal credentials to GitHub. For a deployed Streamlit app, use Streamlit secrets or environment variables for API keys.
