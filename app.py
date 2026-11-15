import io
from datetime import datetime
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Anti-Fragile Stock & Startup Engine",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Anti-Fragile Unified Stock & Startup Evaluation Engine")
st.markdown(
    "Comprehensive evaluation suite featuring the **16-Pillar + Risk Overlay"
    " Framework** for Public Equities and the **Venture Lens** for Pre-Profit"
    " Startups."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration & Mode")
eval_mode = st.sidebar.radio(
    "Select Evaluation Mode",
    ["Public Equity (16-Pillar + Overlays)", "Pre-Profit Startup (Venture Lens)"],
)

if eval_mode == "Public Equity (16-Pillar + Overlays)":
  ticker_input = st.sidebar.text_input(
      "Target Public Ticker", value="NVDA"
  ).strip().upper()
  mos_input = st.sidebar.slider(
      "Margin of Safety Required", 0.05, 0.50, 0.20, 0.05
  )
else:
  st.sidebar.info(
      "💡 **Startup Mode Notice:** Evaluates early-stage pre-profit companies"
      " using venture metrics."
  )
  startup_name = st.sidebar.text_input(
      "Startup Company Name", value="My SaaS Venture"
  ).strip()
  st.sidebar.subheader("Startup Venture Parameters")
  gross_margin_input = st.sidebar.slider(
      "Gross Margin (%)", 10, 95, 75, 5
  ) / 100.0
  rev_growth_input = st.sidebar.slider(
      "YoY ARR Growth Rate (%)", 10, 400, 100, 10
  ) / 100.0
  burn_multiple_input = st.sidebar.slider(
      "Burn Multiple (Burn / Net New ARR)", 0.5, 5.0, 1.3, 0.1
  )
  nrr_input = st.sidebar.slider(
      "Net Revenue Retention (NRR %)", 80, 150, 115, 5
  ) / 100.0
  tam_input = st.sidebar.selectbox(
      "Total Addressable Market (TAM)",
      ["Massive ($10B+)", "Large ($1B-$10B)", "Niche (<$1B)"],
  )
  runway_months = st.sidebar.number_input(
      "Cash Runway (Months)", min_value=1, max_value=60, value=24
  )

run_audit = st.sidebar.button("Run Comprehensive Audit 🚀", type="primary")

if run_audit:
  red_flags = []

  if eval_mode == "Public Equity (16-Pillar + Overlays)":
    if not ticker_input:
      st.error("Please enter a valid ticker symbol.")
      st.stop()

    with st.spinner(
        f"Analyzing public market fundamentals for {ticker_input}..."
    ):
      is_indian = ticker_input.endswith(".NS") or ticker_input.endswith(".BO")
      curr_sym = "₹" if is_indian else "$"
      today_str = datetime.now().strftime("%Y-%m-%d")

      cmp = gm_val = roe_val = de_val = rg_val = fcf_val = net_inc = pe_val = (
          fwd_pe_val
      ) = peg_val = insider_val = fcf_yield_val = None
      company_name = ticker_input
      sector = "Unknown"
      fair_value = 100.0
      analyst_rec = "HOLD"
      num_analysts = 5
      op_margin = 0.15
      eg_val = 0.05
      trailing_eps = 2.0

      try:
        stock = yf.Ticker(ticker_input)
        info = stock.info or {}
        cmp = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
            or 100.0
        )
        company_name = info.get("longName") or info.get("shortName") or ticker_input
        sector = info.get("sector", "Unknown")
        fair_value = info.get("targetMeanPrice") or round(cmp * 1.25, 2)
        analyst_rec = (
            (info.get("recommendationKey") or "HOLD")
            .upper()
            .replace("_", " ")
        )
        num_analysts = info.get("numberOfAnalystOpinions", 5)

        gm_val = info.get("grossMargins", 0.35)
        op_margin = info.get("operatingMargins", 0.12)
        roe_val = info.get("returnOnEquity", 0.10)
        de_val = info.get("debtToEquity", 60.0)
        rg_val = info.get("revenueGrowth", 0.05)
        eg_val = info.get("earningsGrowth", 0.05)

        fcf_val = info.get("freeCashflow", 10_000_000)
        net_inc = info.get("netIncomeToCommon", 10_000_000)

        pe_val = info.get("trailingPE", 25.0)
        fwd_pe_val = info.get("forwardPE", 20.0)
        peg_val = info.get("pegRatio", 1.5)
        trailing_eps = info.get("trailingEps") or (cmp / pe_val if pe_val else 2.0)

        insider_held = info.get("heldPercentInsiders")
        insider_val = (insider_held * 100) if insider_held is not None else 5.0

        mcap = info.get("marketCap", 1)
        fcf_yield_val = (fcf_val / mcap) * 100 if (mcap and mcap > 0 and fcf_val) else 2.5
      except Exception as e:
        st.warning(
            f"Could not fetch live public market data ({e}). Using baseline"
            " estimates."
        )

      # --- RED FLAG DETECTOR FOR PUBLIC EQUITIES ---
      if de_val and de_val > 150:
        red_flags.append(
            f"High Leverage Warning: Debt-to-Equity is {de_val:.1f}%, exceeding"
            " the safe 150% threshold."
        )
      if fcf_val and fcf_val < 0:
        red_flags.append(
            "Negative Free Cash Flow: Company is currently burning cash in"
            " operations."
        )
      if peg_val and peg_val > 2.5:
        red_flags.append(
            f"Valuation Premium Risk: PEG ratio of {peg_val:.2f}x signals overpaid"
            " growth."
        )
      if roe_val and roe_val < 0.05:
        red_flags.append(
            f"Low Capital Efficiency: ROE is {roe_val*100:.1f}%, indicating poor"
            " returns on equity."
        )

      # --- 16 PILLAR SCORING (PUBLIC) ---
      s11 = (
          5
          if gm_val >= 0.50
          else (4 if gm_val >= 0.35 else (3 if gm_val >= 0.20 else 2))
      )
      s12 = (
          5
          if rg_val >= 0.15
          else (4 if rg_val >= 0.08 else (3 if rg_val >= 0.0 else 2))
      )
      s13 = 4
      s21 = (
          5
          if roe_val >= 0.20
          else (4 if roe_val >= 0.12 else (3 if roe_val >= 0.05 else 2))
      )
      s22 = s11
      s23 = 5 if de_val < 50 else (4 if de_val < 150 else (3 if de_val < 300 else 2))
      fcf_ratio = (fcf_val / net_inc) if net_inc and net_inc > 0 else 0.5
      s24 = (
          5
          if fcf_ratio >= 1.0
          else (4 if fcf_ratio >= 0.75 else (3 if fcf_ratio >= 0.5 else 2))
      )
      s31 = (
          5
          if fcf_yield_val >= 7.0
          else (
              4
              if fcf_yield_val >= 4.0
              else (3 if fcf_yield_val >= 2.0 else 2)
          )
      )
      s32 = (
          5
          if (0 < peg_val <= 1.2)
          else (
              4 if (1.2 < peg_val <= 2.0) else (3 if peg_val > 0 else 2)
          )
      )
      s33 = 4
      s34 = (
          5
          if insider_val >= 10.0
          else (4 if insider_val >= 3.0 else (3 if insider_val >= 1.0 else 2))
      )
      s41, s42 = s21, s12
      s43 = 5 if fcf_val > 0 else 3
      s44 = 4
      s45 = (
          5
          if ("BUY" in analyst_rec or "OUTPERFORM" in analyst_rec)
          else (3 if "HOLD" in analyst_rec else 2)
      )

      rows = [
          (
              "1.1",
              "Moat",
              "Competitive Advantage",
              0.06,
              f"Gross Margin: {gm_val*100:.1f}%",
              s11,
              "Pricing power",
          ),
          (
              "1.2",
              "Moat",
              "Market Dynamics",
              0.06,
              f"Rev Growth: {rg_val*100:.1f}%",
              s12,
              "TAM growth",
          ),
          ("1.3", "Moat", "Revenue Model Quality", 0.06, "Recurring", s13, "Retention"),
          (
              "2.1",
              "Quality",
              "ROE / Returns",
              0.07,
              f"ROE: {roe_val*100:.1f}%",
              s21,
              "Efficiency",
          ),
          (
              "2.2",
              "Quality",
              "Gross Margin Floor",
              0.07,
              f"GM: {gm_val*100:.1f}%",
              s22,
              "Stability",
          ),
          (
              "2.3",
              "Quality",
              "Debt Leverage",
              0.07,
              f"D/E: {de_val:.1f}%",
              s23,
              "Solvency",
          ),
          (
              "2.4",
              "Quality",
              "FCF Conversion",
              0.07,
              f"FCF/NI: {fcf_ratio*100:.1f}%",
              s24,
              "Cash quality",
          ),
          (
              "3.1",
              "Valuation",
              "FCF Yield",
              0.07,
              f"Yield: {fcf_yield_val:.1f}%",
              s31,
              "Cash yield",
          ),
          (
              "3.2",
              "Valuation",
              "PEG Multiple",
              0.06,
              f"PEG: {peg_val:.2f}x",
              s32,
              "Growth multiple",
          ),
          (
              "3.3",
              "Governance",
              "Alignment",
              0.06,
              "Shareholder returns",
              s33,
              "Capital policy",
          ),
          (
              "3.4",
              "Governance",
              "Insider Ownership",
              0.05,
              f"Insiders: {insider_val:.1f}%",
              s34,
              "Skin in game",
          ),
          (
              "4.1",
              "Filters",
              "Multi-Yr Return",
              0.06,
              f"ROE: {roe_val*100:.1f}%",
              s41,
              "Durability",
          ),
          (
              "4.2",
              "Filters",
              "Top-Line Compounding",
              0.06,
              f"Rev Growth: {rg_val*100:.1f}%",
              s42,
              "Expansion",
          ),
          (
              "4.3",
              "Filters",
              "Positive FCF",
              0.06,
              f"FCF: {curr_sym}{fcf_val:,.0f}",
              s43,
              "Self-funding",
          ),
          ("4.4", "Filters", "Insider Activity", 0.06, "Stable", s44, "Net action"),
          (
              "4.5",
              "Consensus",
              "Analyst Rating",
              0.06,
              f"{analyst_rec}",
              s45,
              "Analyst view",
          ),
      ]
      base_score = sum(wt * sc for _, _, _, wt, _, sc, _ in rows)

      eps_rev_pct = eg_val if eg_val < -0.05 else 0.0
      bookings_delta = eg_val - rg_val
      val_cap_flag = (
          1 if (fwd_pe_val and fwd_pe_val >= 45.0 and peg_val > 2.5) else 0
      )
      cyclical_ratio = (
          round(op_margin / 0.15, 2)
          if sector in {"Energy", "Basic Materials", "Industrials"}
          else 1.0
      )

      penalties = (
          (-0.50 if eps_rev_pct < -0.05 else 0.0)
          + (-0.40 if bookings_delta < 0.0 else 0.0)
          + (-0.60 if val_cap_flag >= 1 else 0.0)
          + (-0.50 if cyclical_ratio > 1.50 else 0.0)
      )
      final_score = max(1.0, min(5.0, base_score + penalties))

      if val_cap_flag >= 1:
        verdict = "NEUTRAL / VAL CAP"
      elif final_score >= 4.0:
        verdict = "PASS / BUY"
      elif final_score >= 3.2:
        verdict = "WATCH / HOLD"
      else:
        verdict = "FAIL / AVOID"

      target_buy_price = fair_value * (1 - mos_input)

      # --- YEAR 1 & YEAR 2 FORECAST MODEL (PEG & PROFIT GROWTH) ---
      # Use earnings growth (or default to 10% if missing)
      g_rate = eg_val if eg_val and -0.8 < eg_val < 2.0 else 0.12
      
      # Projected EPS & Estimated Profit (Net Income)
      est_net_inc_yr1 = net_inc * (1 + g_rate)
      est_net_inc_yr2 = est_net_inc_yr1 * (1 + g_rate)
      
      proj_eps_yr1 = trailing_eps * (1 + g_rate)
      proj_eps_yr2 = proj_eps_yr1 * (1 + g_rate)
      
      # Implied P/E multiplier derived from PEG ratio & growth rate
      # PEG = (P/E) / GrowthRate -> Target P/E = PEG * (GrowthRate * 100)
      base_peg = peg_val if peg_val and peg_val > 0 else 1.5
      implied_pe = base_peg * (max(g_rate, 0.05) * 100)
      safe_pe = min(max(implied_pe, 8.0), 50.0)  # Bound P/E between 8x and 50x
      
      forecast_price_yr1 = proj_eps_yr1 * safe_pe
      forecast_price_yr2 = proj_eps_yr2 * safe_pe

  else:
    # --- PRE-PROFIT STARTUP / VENTURE LENS ---
    company_name = startup_name
    curr_sym = "$"

    if runway_months < 12:
      red_flags.append(
          f"Critical Runway Risk: Only {runway_months} months of cash runway"
          " remaining."
      )
    if burn_multiple_input > 2.5:
      red_flags.append(
          f"Inefficient Burn: Burn Multiple is {burn_multiple_input}x (burning"
          " >$2.50 for every $1 of new ARR)."
      )
    if gross_margin_input < 0.50:
      red_flags.append(
          f"Low Gross Margin: {gross_margin_input*100:.1f}% fails scalable unit"
          " economic hurdles."
      )

    s11 = 5 if gross_margin_input >= 0.75 else (4 if gross_margin_input >= 0.60 else 2)
    s12 = 5 if rev_growth_input >= 1.0 else (4 if rev_growth_input >= 0.50 else 2)
    s13 = 5 if burn_multiple_input <= 1.2 else (3 if burn_multiple_input <= 2.0 else 1)
    s14 = 5 if nrr_input >= 1.20 else (4 if nrr_input >= 1.05 else 2)
    s15 = 5 if tam_input == "Massive ($10B+)" else (3 if tam_input == "Large ($1B-$10B)" else 1)
    s16 = 5 if runway_months >= 24 else (3 if runway_months >= 12 else 1)

    rows = [
        (
            "S.1",
            "Unit Economics",
            "Gross Margin Profile",
            0.15,
            f"GM: {gross_margin_input*100:.1f}%",
            s11,
            "Scalability",
        ),
        (
            "S.2",
            "Growth",
            "Top-Line ARR Growth",
            0.20,
            f"YoY: {rev_growth_input*100:.1f}%",
            s12,
            "Traction velocity",
        ),
        (
            "S.3",
            "Efficiency",
            "Burn Multiple",
            0.20,
            f"Burn Mult: {burn_multiple_input}x",
            s13,
            "Capital efficiency",
        ),
        (
            "S.4",
            "Retention",
            "Net Revenue Retention",
            0.15,
            f"NRR: {nrr_input*100:.1f}%",
            s14,
            "Product stickiness",
        ),
        (
            "S.5",
            "Market",
            "Addressable Market (TAM)",
            0.15,
            f"{tam_input}",
            s15,
            "Market ceiling",
        ),
        (
            "S.6",
            "Risk",
            "Cash Runway Horizon",
            0.15,
            f"Runway: {runway_months} mos",
            s16,
            "Survival buffer",
        ),
    ]
    final_score = sum(wt * sc for _, _, _, wt, _, sc, _ in rows)
    verdict = (
        "HIGH-POTENTIAL VENTURE"
        if final_score >= 4.0
        else ("WATCHLIST STARTUP" if final_score >= 3.0 else "HIGH BURN RISK")
    )

  # --- RENDER DASHBOARD ---
  st.subheader(f"Evaluation Dashboard: {company_name}")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Evaluation Mode", eval_mode.split("(")[0])
  col2.metric("Composite Score", f"{final_score:.2f} / 5.00")
  if eval_mode.startswith("Public"):
    col3.metric("CMP", f"{curr_sym}{cmp:,.2f}")
    col4.metric(
        "Target Max Buy Price",
        f"{curr_sym}{target_buy_price:,.2f}",
        delta="IN BUY ZONE" if cmp <= target_buy_price else "WAIT FOR DIP",
        delta_color="normal" if cmp <= target_buy_price else "inverse",
    )
  else:
    col3.metric("Cash Runway", f"{runway_months} Months")
    col4.metric("Burn Multiple", f"{burn_multiple_input}x")

  # --- YEAR 1 & YEAR 2 FORECAST METRICS SECTION (PUBLIC MODE) ---
  if eval_mode.startswith("Public"):
    st.markdown("### 📈 Year 1 & Year 2 Forecast Model (PEG & Profit Growth)")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric(
        "Year 1 Forecast Price",
        f"{curr_sym}{forecast_price_yr1:,.2f}",
        delta=f"{((forecast_price_yr1/cmp)-1)*100:+.1f}% vs CMP",
    )
    fc2.metric(
        "Year 2 Forecast Price",
        f"{curr_sym}{forecast_price_yr2:,.2f}",
        delta=f"{((forecast_price_yr2/cmp)-1)*100:+.1f}% vs CMP",
    )
    fc3.metric("Est. Net Profit (Yr 1)", f"{curr_sym}{est_net_inc_yr1:,.0f}")
    fc4.metric("Est. Net Profit (Yr 2)", f"{curr_sym}{est_net_inc_yr2:,.0f}")

  # Verdict Banner
  if "PASS" in verdict or "VENTURE" in verdict:
    st.success(f"**Verdict:** {verdict}")
  elif "WATCH" in verdict:
    st.warning(f"**Verdict:** {verdict}")
  else:
    st.error(f"**Verdict:** {verdict}")

  # --- RED FLAG ALERT BOX ---
  if red_flags:
    st.error("🚨 **Automated Red Flags Detected:**")
    for flag in red_flags:
      st.markdown(f"- {flag}")
  else:
    st.success("✅ **Clean Audit:** No critical structural red flags triggered.")

  # --- INTERACTIVE PLOTLY VISUALIZATION (FIXED FONT & HEIGHT) ---
  st.markdown("### 📊 Visual Score Breakdown")
  df_chart = pd.DataFrame(
      [
          {
              "Ref": r[0],
              "Category": r[1],
              "Criteria": r[2],
              "Score": r[5],
              "Metric": r[4],
          }
          for r in rows
      ]
  )
  fig = px.bar(
      df_chart,
      x="Score",
      y="Criteria",
      orientation="h",
      color="Score",
      color_continuous_scale="Teal",
      range_x=[0, 5],
      text="Metric",
      title="Pillar Score Distribution (Scale 1 to 5)",
  )
  # Updated trace and layout settings to fix font sizes and spacing
  fig.update_traces(
      textfont_size=13, textangle=0, textposition="inside", cliponaxis=False
  )
  fig.update_layout(
      yaxis={"categoryorder": "total ascending"},
      height=650,  # Increased height so text labels never overlap
      font=dict(size=14, color="#1E293B"),
      margin=dict(l=20, r=20, t=50, b=20),
  )
  st.plotly_chart(fig, use_container_width=True)

  # Detailed Pillar Scorecard Table
  st.markdown("### Detailed Pillar Scorecard")
  df_display = pd.DataFrame(
      [
          {
              "Ref": r[0],
              "Category": r[1],
              "Pillar / Criteria": r[2],
              "Weight": f"{r[3]*100:.0f}%",
              "Metric": r[4],
              "Score": r[5],
              "Notes": r[6],
          }
          for r in rows
      ]
  )
  st.dataframe(df_display, use_container_width=True, hide_index=True)

  # --- EDUCATIONAL REFERENCES & EXPLANATIONS ---
  with st.expander("📚 Educational Reference & Glossary for Key Metrics"):
    st.markdown("""
        To learn more about how these financial concepts are evaluated in institutional analysis, explore these references:
        * **[Investopedia: Margin of Safety](https://www.investopedia.com/terms/m/marginofsafety.asp)** — Explains the principle of purchasing assets below their intrinsic fair value to mitigate downside risk.
        * **[Bessemer Venture Partners: Burn Multiple](https://www.bvp.com/atlas/the-burn-multiple)** — A premier metric for evaluating how efficiently a pre-profit startup turns capital burn into net new Annual Recurring Revenue (ARR).
        * **[Corporate Finance Institute: Debt-to-Equity Ratio](https://corporatefinanceinstitute.com/resources/commercial-lending/debt-to-equity-ratio/)** — Measures financial leverage and long-term solvency risk.
        * **[Sifted: Net Revenue Retention (NRR)](
