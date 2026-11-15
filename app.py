from datetime import datetime
import math
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Anti-Fragile Equity & Venture Screening Engine",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Anti-Fragile Equity & Venture Screening Engine (v2.0)")
st.markdown(
    "Institutional-style quantitative screening framework featuring"
    " **non-compensatory knockout gates**, **multi-year fundamental checks**,"
    " and **strict data transparency**."
)


# --- HELPER FUNCTIONS ---
def safe_num(val):
  """Safely extracts a valid float, returning None if missing, None, or NaN."""
  if val is None:
    return None
  try:
    f = float(val)
    if math.isnan(f) or math.isinf(f):
      return None
    return f
  except (ValueError, TypeError):
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_bundle(ticker_symbol):
  """Fetches and caches info, financials, and balance sheet from yfinance."""
  tk = yf.Ticker(ticker_symbol)
  info = tk.info or {}
  try:
    fin = tk.financials
  except Exception:
    fin = pd.DataFrame()
  try:
    bs = tk.balance_sheet
  except Exception:
    bs = pd.DataFrame()
  try:
    cf = tk.cashflow
  except Exception:
    cf = pd.DataFrame()
  return {
      "info": info,
      "financials": fin,
      "balance_sheet": bs,
      "cashflow": cf,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  }


def get_statement_row(df, possible_names):
  """Extracts a row series (most recent to oldest) from a financial statement."""
  if df is None or df.empty:
    return []
  for name in possible_names:
    if name in df.index:
      vals = [safe_num(v) for v in df.loc[name].tolist()]
      return [v for v in vals if v is not None]
  return []


def compute_cagr(series_newest_first):
  """Computes CAGR from a newest-first list of annual values."""
  if len(series_newest_first) < 2:
    return None
  end_val = series_newest_first[0]
  start_val = series_newest_first[-1]
  years = len(series_newest_first) - 1
  if start_val is None or end_val is None or start_val <= 0 or end_val <= 0:
    return None
  return (end_val / start_val) ** (1.0 / years) - 1.0


# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration & Mode")
eval_mode = st.sidebar.radio(
    "Select Evaluation Lens",
    [
        "Public Equity (Quality & Solvency Lens)",
        "Pre-Profit / High-Growth (Venture Lens)",
    ],
)

if eval_mode.startswith("Public"):
  ticker_input = st.sidebar.text_input(
      "Target Public Ticker", value="MSFT"
  ).strip().upper()
  mos_input = st.sidebar.slider(
      "Discount Applied to Analyst Consensus (%)", 5, 50, 20, 5
  ) / 100.0
else:
  st.sidebar.info(
      "💡 **Venture Lens:** Evaluates cash runway, burn efficiency, and"
      " breakeven trajectory for pre-profit businesses."
  )
  input_source = st.sidebar.radio(
      "Venture Data Source",
      ["Analyze Public Ticker (Live Statements)", "Manual Startup Simulation"],
  )
  if input_source == "Analyze Public Ticker (Live Statements)":
    startup_ticker_input = st.sidebar.text_input(
        "Enter Ticker", value="CRWV"
    ).strip().upper()
  else:
    startup_name = st.sidebar.text_input(
        "Startup Name", value="Simulated SaaS Venture"
    ).strip()
    st.sidebar.subheader("Unit Economics & Balance Sheet")
    sim_arr = (
        st.sidebar.number_input(
            "Current Annual Revenue / ARR ($M)", min_value=0.1, value=10.0, step=1.0
        )
        * 1_000_000
    )
    sim_cash = (
        st.sidebar.number_input(
            "Cash on Balance Sheet ($M)", min_value=0.1, value=24.0, step=1.0
        )
        * 1_000_000
    )
    gross_margin_input = st.sidebar.slider(
        "Gross Margin (%)", 10, 95, 75, 5
    ) / 100.0
    rev_growth_input = st.sidebar.slider(
        "YoY Revenue Growth Rate (%)", 10, 300, 80, 5
    ) / 100.0
    burn_multiple_input = st.sidebar.slider(
        "Burn Multiple (Net Burn / Net New ARR)", 0.2, 5.0, 1.3, 0.1
    )
    nrr_input = st.sidebar.slider(
        "Net Revenue Retention (NRR %)", 70, 160, 115, 5
    ) / 100.0
    tam_input = st.sidebar.selectbox(
        "Total Addressable Market (TAM)",
        ["Massive ($10B+)", "Large ($1B-$10B)", "Niche (<$1B)"],
    )

run_audit = st.sidebar.button("Run Quantitative Screen 🚀", type="primary")

if run_audit:
  knockout_flags = []
  warning_flags = []

  # =========================================================================
  # BRANCH 1: PUBLIC EQUITY (QUALITY, SOLVENCY & VALUATION LENS)
  # =========================================================================
  if eval_mode.startswith("Public"):
    if not ticker_input:
      st.error("Please enter a valid ticker symbol.")
      st.stop()

    with st.spinner(f"Fetching verified statements for {ticker_input}..."):
      bundle = fetch_ticker_bundle(ticker_input)
      info = bundle["info"]
      fin = bundle["financials"]
      bs = bundle["balance_sheet"]
      cf = bundle["cashflow"]
      fetch_time = bundle["timestamp"]

    cmp = safe_num(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    if cmp is None:
      st.error(
          f"Unable to retrieve live pricing data for `{ticker_input}`. Verify"
          " the ticker symbol (e.g., `RELIANCE.NS` for NSE, `MSFT` for US)."
      )
      st.stop()

    company_name = info.get("longName") or info.get("shortName") or ticker_input
    sector = info.get("sector") or "Unknown"
    industry = info.get("industry") or "Unknown"
    is_indian = ticker_input.endswith(".NS") or ticker_input.endswith(".BO")
    curr_sym = "₹" if is_indian else "$"

    # Sector Exclusion Warning
    excluded_sectors = {"Financial Services", "Real Estate", "Utilities"}
    if sector in excluded_sectors:
      st.warning(
          f"⚠️ **Sector Rubric Notice ({sector} / {industry}):** Traditional"
          " Gross Margin and Debt-to-Equity thresholds do not apply cleanly to"
          " banks, NBFCs, insurers, REITs, or regulated utilities. Interpret"
          " leverage and margin scores with caution."
      )

    # Extract Core Metrics (Strictly None if missing)
    gm_val = safe_num(info.get("grossMargins"))
    op_margin = safe_num(info.get("operatingMargins"))
    roe_val = safe_num(info.get("returnOnEquity"))
    roa_val = safe_num(info.get("returnOnAssets"))
    de_val = safe_num(info.get("debtToEquity"))
    rg_val = safe_num(info.get("revenueGrowth"))
    eg_val = safe_num(info.get("earningsGrowth"))
    fcf_val = safe_num(info.get("freeCashflow"))
    net_inc = safe_num(info.get("netIncomeToCommon"))
    pe_val = safe_num(info.get("trailingPE"))
    fwd_pe_val = safe_num(info.get("forwardPE"))
    peg_val = safe_num(info.get("pegRatio"))
    trailing_eps = safe_num(info.get("trailingEps"))
    mcap = safe_num(info.get("marketCap"))
    insider_held = safe_num(info.get("heldPercentInsiders"))
    insider_val = (insider_held * 100.0) if insider_held is not None else None
    target_mean = safe_num(info.get("targetMeanPrice"))
    analyst_rec = (
        (info.get("recommendationKey") or "").upper().replace("_", " ") or None
    )

    # Multi-Year Historical Statement Extraction
    rev_hist = get_statement_row(fin, ["Total Revenue", "Operating Revenue"])
    ebit_hist = get_statement_row(fin, ["EBIT", "Operating Income"])
    int_hist = get_statement_row(
        fin, ["Interest Expense", "Interest Expense Non Operating"]
    )
    fcf_hist = get_statement_row(cf, ["Free Cash Flow"])
    equity_hist = get_statement_row(
        bs, ["Stockholders Equity", "Total Stockholder Equity"]
    )
    shares_hist = get_statement_row(
        bs, ["Ordinary Shares Number", "Share Issued"]
    )

    rev_cagr = compute_cagr(rev_hist)
    interest_coverage = None
    if ebit_hist and int_hist and int_hist[0] and abs(int_hist[0]) > 0:
      interest_coverage = ebit_hist[0] / abs(int_hist[0])

    share_dilution_pct = None
    if len(shares_hist) >= 2 and shares_hist[1] and shares_hist[1] > 0:
      share_dilution_pct = ((shares_hist[0] / shares_hist[1]) - 1.0) * 100.0

    neg_equity = (
        True
        if (equity_hist and equity_hist[0] is not None and equity_hist[0] < 0)
        else False
    )

    fcf_yield_val = (
        (fcf_val / mcap) * 100.0
        if (fcf_val is not None and mcap is not None and mcap > 0)
        else None
    )
    fcf_conversion = (
        (fcf_val / net_inc)
        if (fcf_val is not None and net_inc is not None and net_inc > 0)
        else None
    )

    # Smart Auto-Routing Check: Only route if unprofitable AND negative FCF AND high top-line growth
    if (
        net_inc is not None
        and net_inc < 0
        and fcf_val is not None
        and fcf_val < 0
        and rg_val is not None
        and rg_val > 0.25
    ):
      st.info(
          f"ℹ️ **High-Growth Pre-Profit Profile Detected:** `{company_name}` is"
          " growing top-line revenue >25% while burning cash. Consider"
          " switching to the **Pre-Profit / High-Growth (Venture Lens)** in the"
          " sidebar for runway and burn-efficiency analysis."
      )

    # --- NON-COMPENSATORY KNOCKOUT GATES & WARNINGS ---
    if neg_equity:
      knockout_flags.append(
          "Negative Shareholder Equity detected on balance sheet (D/E ratio"
          " distorted)."
      )
    elif de_val is not None and de_val > 250.0 and sector not in excluded_sectors:
      knockout_flags.append(
          f"Extreme Leverage Knockout: Debt-to-Equity is {de_val:.1f}% (>250%"
          " threshold)."
      )
    elif de_val is not None and de_val > 150.0 and sector not in excluded_sectors:
      warning_flags.append(
          f"Elevated Leverage: Debt-to-Equity is {de_val:.1f}% (>150% threshold)."
      )

    if (
        interest_coverage is not None
        and interest_coverage < 2.0
        and sector not in excluded_sectors
    ):
      knockout_flags.append(
          f"Debt Service Fragility: Interest Coverage is {interest_coverage:.2f}x"
          " (<2.0x minimum safety floor)."
      )

    if len(fcf_hist) >= 2 and fcf_hist[0] < 0 and fcf_hist[1] < 0:
      knockout_flags.append(
          "Consecutive Negative Free Cash Flow across the last 2 fiscal years."
      )
    elif fcf_val is not None and fcf_val < 0:
      warning_flags.append(
          f"Trailing Negative Free Cash Flow ({curr_sym}{fcf_val:,.0f})."
      )

    if peg_val is not None and peg_val > 2.8:
      warning_flags.append(
          f"Valuation Stretch: PEG Ratio is {peg_val:.2f}x (>2.8x)."
      )
    if share_dilution_pct is not None and share_dilution_pct > 5.0:
      warning_flags.append(
          f"Shareholder Dilution: Share count expanded {share_dilution_pct:.1f}%"
          " YoY."
      )

    # --- INDEPENDENT SIGNAL SCORING (NO DUPLICATES, NO HARDCODED 4/5s) ---
    # Each tuple: (Ref, Category, Pillar, Weight, Metric_Display, Score_or_None, Notes)
    pillars = []

    # 1. Gross Margin (Pricing Power)
    s_gm = (
        (
            5
            if gm_val >= 0.55
            else (4 if gm_val >= 0.40 else (3 if gm_val >= 0.25 else (2 if gm_val >= 0.15 else 1)))
        )
        if gm_val is not None
        else None
    )
    pillars.append((
        "1.1",
        "Moat",
        "Gross Margin (Pricing Power)",
        0.10,
        f"{gm_val*100:.1f}%" if gm_val is not None else "N/A",
        s_gm,
        "Unit margin durability",
    ))

    # 2. Multi-Year Revenue CAGR (or YoY if history unavailable)
    growth_metric = rev_cagr if rev_cagr is not None else rg_val
    growth_label = (
        f"{rev_cagr*100:.1f}% (Multi-Yr CAGR)"
        if rev_cagr is not None
        else (f"{rg_val*100:.1f}% (YoY)" if rg_val is not None else "N/A")
    )
    s_rev = (
        (
            5
            if growth_metric >= 0.18
            else (4 if growth_metric >= 0.10 else (3 if growth_metric >= 0.04 else (2 if growth_metric >= 0.0 else 1)))
        )
        if growth_metric is not None
        else None
    )
    pillars.append((
        "1.2",
        "Growth",
        "Top-Line Compounding",
        0.10,
        growth_label,
        s_rev,
        "Organic expansion",
    ))

    # 3. Operating Margin
    s_opm = (
        (
            5
            if op_margin >= 0.25
            else (4 if op_margin >= 0.15 else (3 if op_margin >= 0.08 else (2 if op_margin > 0 else 1)))
        )
        if op_margin is not None
        else None
    )
    pillars.append((
        "1.3",
        "Moat",
        "Operating Margin",
        0.08,
        f"{op_margin*100:.1f}%" if op_margin is not None else "N/A",
        s_opm,
        "Operating leverage",
    ))

    # 4. Return on Equity (Capital Efficiency)
    s_roe = (
        (
            5
            if roe_val >= 0.22
            else (4 if roe_val >= 0.14 else (3 if roe_val >= 0.08 else (2 if roe_val > 0 else 1)))
        )
        if (roe_val is not None and not neg_equity)
        else None
    )
    pillars.append((
        "2.1",
        "Quality",
        "Return on Equity (ROE)",
        0.10,
        f"{roe_val*100:.1f}%"
        if (roe_val is not None and not neg_equity)
        else ("Neg. Equity" if neg_equity else "N/A"),
        s_roe,
        "Capital return",
    ))

    # 5. Return on Assets (Unlevered Efficiency)
    s_roa = (
        (
            5
            if roa_val >= 0.12
            else (4 if roa_val >= 0.07 else (3 if roa_val >= 0.03 else (2 if roa_val > 0 else 1)))
        )
        if roa_val is not None
        else None
    )
    pillars.append((
        "2.2",
        "Quality",
        "Return on Assets (ROA)",
        0.08,
        f"{roa_val*100:.1f}%" if roa_val is not None else "N/A",
        s_roa,
        "Unlevered asset yield",
    ))

    # 6. Debt-to-Equity Solvency
    if neg_equity:
      s_de = 1
      de_disp = "Neg. Equity"
    elif de_val is not None:
      s_de = (
          5
          if de_val < 40
          else (4 if de_val < 100 else (3 if de_val < 180 else (2 if de_val < 250 else 1)))
      )
      de_disp = f"{de_val:.1f}%"
    else:
      s_de = None
      de_disp = "N/A"
    pillars.append((
        "2.3",
        "Solvency",
        "Debt-to-Equity Ratio",
        0.10,
        de_disp,
        s_de,
        "Balance sheet leverage",
    ))

    # 7. Interest Coverage Ratio
    s_ic = (
        (
            5
            if interest_coverage >= 10.0
            else (4 if interest_coverage >= 5.0 else (3 if interest_coverage >= 2.5 else (2 if interest_coverage >= 1.5 else 1)))
        )
        if interest_coverage is not None
        else None
    )
    pillars.append((
        "2.4",
        "Solvency",
        "Interest Coverage (EBIT/Int)",
        0.08,
        f"{interest_coverage:.1f}x" if interest_coverage is not None else "N/A",
        s_ic,
        "Debt servicing safety",
    ))

    # 8. FCF Conversion (Cash Quality)
    s_fcf_conv = (
        (
            5
            if fcf_conversion >= 1.0
            else (4 if fcf_conversion >= 0.75 else (3 if fcf_conversion >= 0.45 else (2 if fcf_conversion > 0 else 1)))
        )
        if fcf_conversion is not None
        else None
    )
    pillars.append((
        "2.5",
        "Quality",
        "FCF Conversion (FCF / Net Inc)",
        0.08,
        f"{fcf_conversion*100:.1f}%" if fcf_conversion is not None else "N/A",
        s_fcf_conv,
        "Earnings cash backing",
    ))

    # 9. Free Cash Flow Yield
    s_fcf_y = (
        (
            5
            if fcf_yield_val >= 6.5
            else (4 if fcf_yield_val >= 4.0 else (3 if fcf_yield_val >= 2.0 else (2 if fcf_yield_val > 0 else 1)))
        )
        if fcf_yield_val is not None
        else None
    )
    pillars.append((
        "3.1",
        "Valuation",
        "Free Cash Flow Yield",
        0.08,
        f"{fcf_yield_val:.2f}%" if fcf_yield_val is not None else "N/A",
        s_fcf_y,
        "Cash yield vs Market Cap",
    ))

    # 10. PEG Ratio
    s_peg = (
        (
            5
            if (0 < peg_val <= 1.1)
            else (4 if (1.1 < peg_val <= 1.8) else (3 if (1.8 < peg_val <= 2.5) else 2))
        )
        if (peg_val is not None and peg_val > 0)
        else None
    )
    pillars.append((
        "3.2",
        "Valuation",
        "PEG Multiple",
        0.08,
        f"{peg_val:.2f}x" if (peg_val is not None and peg_val > 0) else "N/A",
        s_peg,
        "Growth-adjusted P/E",
    ))

    # 11. Share Dilution Discipline
    s_dil = (
        (
            5
            if share_dilution_pct <= 0.0
            else (4 if share_dilution_pct <= 2.0 else (3 if share_dilution_pct <= 5.0 else 1))
        )
        if share_dilution_pct is not None
        else None
    )
    pillars.append((
        "4.1",
        "Governance",
        "YoY Share Count Change",
        0.06,
        f"{share_dilution_pct:+.1f}%" if share_dilution_pct is not None else "N/A",
        s_dil,
        "Buybacks vs dilution",
    ))

    # 12. Insider Ownership
    s_ins = (
        (
            5
            if insider_val >= 10.0
            else (4 if insider_val >= 3.0 else (3 if insider_val >= 0.5 else 2))
        )
        if insider_val is not None
        else None
    )
    pillars.append((
        "4.2",
        "Governance",
        "Insider Ownership",
        0.06,
        f"{insider_val:.1f}%" if insider_val is not None else "N/A",
        s_ins,
        "Management alignment",
    ))

    # --- DATA COMPLETENESS & DYNAMIC WEIGHT RE-NORMALIZATION ---
    valid_pillars = [p for p in pillars if p[5] is not None]
    total_metrics = len(pillars)
    sourced_metrics = len(valid_pillars)
    coverage_pct = (sourced_metrics / total_metrics) * 100.0

    if sourced_metrics > 0:
      total_valid_weight = sum(p[3] for p in valid_pillars)
      final_score = sum((p[3] / total_valid_weight) * p[5] for p in valid_pillars)
    else:
      final_score = 0.0

    # --- SCREENING VERDICT WITH KNOCKOUT CAPS ---
    if coverage_pct < 60.0:
      verdict = "INSUFFICIENT DATA — SCREEN REFUSED"
      verdict_style = "warning"
    elif knockout_flags:
      verdict = "FRAGILE / HIGH RISK (KNOCKOUT TRIGGERED)"
      verdict_style = "error"
    elif final_score >= 4.0:
      verdict = "SCREEN RESULT: STRONG QUALITY"
      verdict_style = "success"
    elif final_score >= 3.1:
      verdict = "SCREEN RESULT: MIXED / WATCHLIST"
      verdict_style = "warning"
    else:
      verdict = "SCREEN RESULT: WEAK FUNDAMENTALS"
      verdict_style = "error"

    # --- ANALYST TARGET & FORWARD PROJECTION (ONLY FOR PROFITABLE COMPANIES) ---
    discounted_target = (
        target_mean * (1.0 - mos_input) if target_mean is not None else None
    )

    can_project_eps = (
        trailing_eps is not None
        and trailing_eps > 0
        and net_inc is not None
        and net_inc > 0
    )
    if can_project_eps:
      raw_g = eg_val if (eg_val is not None and -0.5 < eg_val < 1.5) else (rev_cagr or 0.08)
      g_yr1 = max(min(raw_g, 0.40), -0.20)
      g_yr2 = g_yr1 * 0.80  # Realistic growth decay in Year 2

      proj_net_inc_yr1 = net_inc * (1.0 + g_yr1)
      proj_net_inc_yr2 = proj_net_inc_yr1 * (1.0 + g_yr2)

      proj_eps_yr1 = trailing_eps * (1.0 + g_yr1)
      proj_eps_yr2 = proj_eps_yr1 * (1.0 + g_yr2)

      ref_pe = (
          fwd_pe_val
          if (fwd_pe_val is not None and 5.0 <= fwd_pe_val <= 45.0)
          else (
              pe_val
              if (pe_val is not None and 5.0 <= pe_val <= 45.0)
              else 18.0
          )
      )
      forecast_price_yr1 = proj_eps_yr1 * ref_pe
      forecast_price_yr2 = proj_eps_yr2 * ref_pe

    # --- RENDER PUBLIC EQUITY DASHBOARD ---
    st.subheader(f"Quantitative Screen: {company_name} ({ticker_input})")
    st.caption(
        f"Sector: {sector} | Industry: {industry} | Data Timestamp:"
        f" {fetch_time} | Data Completeness: **{sourced_metrics} /"
        f" {total_metrics} Metrics Sourced ({coverage_pct:.0f}%)**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Score", f"{final_score:.2f} / 5.00" if coverage_pct >= 60 else "N/A")
    c2.metric("Current Market Price", f"{curr_sym}{cmp:,.2f}")

    if discounted_target is not None:
      in_zone = cmp <= discounted_target
      c3.metric(
          f"Discounted Consensus ({int(mos_input*100)}% Off)",
          f"{curr_sym}{discounted_target:,.2f}",
          delta="BELOW DISCOUNTED TARGET" if in_zone else "ABOVE DISCOUNTED TARGET",
          delta_color="normal" if in_zone else "inverse",
      )
    else:
      c3.metric("Analyst Consensus Target", "N/A (No Coverage)")

    c4.metric(
        "Data Completeness",
        f"{sourced_metrics}/{total_metrics} Sourced",
        delta="PASS" if coverage_pct >= 60 else "LOW COVERAGE",
        delta_color="normal" if coverage_pct >= 60 else "inverse",
    )

    # Verdict Banner
    if verdict_style == "success":
      st.success(f"**{verdict}**")
    elif verdict_style == "warning":
      st.warning(f"**{verdict}**")
    else:
      st.error(f"**{verdict}**")

    # Knockout & Warning Flags
    if knockout_flags:
      st.error("🛑 **Non-Compensatory Knockout Flags (Caps Verdict):**")
      for kf in knockout_flags:
        st.markdown(f"- {kf}")
    if warning_flags:
      st.warning("⚠️ **Structural Risk Warnings:**")
      for wf in warning_flags:
        st.markdown(f"- {wf}")
    if not knockout_flags and not warning_flags and coverage_pct >= 60:
      st.success("✅ **Clean Risk Audit:** Zero knockout or structural warnings triggered.")

    # Forward Valuation Section (Profitable Companies Only)
    st.markdown("### 📈 Illustrative Year 1 & Year 2 Earnings & Price Trajectory")
    if can_project_eps:
      fc1, fc2, fc3, fc4 = st.columns(4)
      fc1.metric(
          "Year 1 Implied Price",
          f"{curr_sym}{forecast_price_yr1:,.2f}",
          delta=f"{((forecast_price_yr1/cmp)-1)*100:+.1f}% vs CMP",
      )
      fc2.metric(
          "Year 2 Implied Price",
          f"{curr_sym}{forecast_price_yr2:,.2f}",
          delta=f"{((forecast_price_yr2/cmp)-1)*100:+.1f}% vs CMP",
      )
      
