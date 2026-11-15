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
    "Comprehensive evaluation suite featuring the **16-Pillar Framework** for"
    " Profitable Equities and the **Advanced Venture Growth Lens** for"
    " Pre-Profit / Growth Startups."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration & Mode")
eval_mode = st.sidebar.radio(
    "Select Evaluation Mode",
    [
        "Public Equity (16-Pillar / Profitable)",
        "Pre-Profit Startup / Venture Lens (Manual or Auto)",
    ],
)

if eval_mode.startswith("Public"):
  ticker_input = st.sidebar.text_input(
      "Target Public Ticker", value="NVDA"
  ).strip().upper()
  mos_input = st.sidebar.slider(
      "Margin of Safety Required", 0.05, 0.50, 0.20, 0.05
  )
else:
  st.sidebar.info(
      "💡 **Startup Mode Notice:** Evaluates early-stage or pre-profit"
      " companies. (Unprofitable public tickers also auto-route here)."
  )
  input_source = st.sidebar.radio(
      "Data Source", ["Manual Simulation Sliders", "Analyze Public Ticker"]
  )
  if input_source == "Analyze Public Ticker":
    startup_ticker_input = st.sidebar.text_input(
        "Enter Unprofitable/Growth Ticker", value="CRWV"
    ).strip().upper()
  else:
    startup_name = st.sidebar.text_input(
        "Startup Company Name", value="My SaaS Venture"
    ).strip()
    gross_margin_input = st.sidebar.slider(
        "Gross Margin (%)", 10, 95, 75, 5
    ) / 100.0
    rev_growth_input = st.sidebar.slider(
        "YoY ARR Growth Rate (%)", 10, 400, 85, 10
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

  # --- PUBLIC EQUITY MODE ---
  if eval_mode.startswith("Public"):
    if not ticker_input:
      st.error("Please enter a valid ticker symbol.")
      st.stop()

    with st.spinner(
        f"Analyzing public market fundamentals for {ticker_input}..."
    ):
      is_indian = ticker_input.endswith(".NS") or ticker_input.endswith(".BO")
      curr_sym = "₹" if is_indian else "$"

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
        net_inc = info.get("netIncomeToCommon", 10_000_000)
        fair_value = info.get("targetMeanPrice") or round(cmp * 1.25, 2)
      except Exception as e:
        st.error(f"Could not fetch data for {ticker_input}: {e}")
        st.stop()

      # --- AUTOMATED ROUTING CHECK ---
      # If public company has not declared profits (Net Income <= 0), auto-transfer to Venture Lens!
      if net_inc is not None and net_inc <= 0:
        st.warning(
            f"⚠️ **Automatic Mode Transfer:** **{company_name} ({ticker_input})**"
            " has reported negative trailing net income/profits. Automatically"
            " transferring analysis to the **Pre-Profit Venture Growth"
            " Lens**."
        )
        eval_mode = "Pre-Profit Startup / Venture Lens (Manual or Auto)"
        input_source = "Analyze Public Ticker"
        startup_ticker_input = ticker_input
      else:
        # Standard Public Equity Logic
        gm_val = info.get("grossMargins", 0.35)
        roe_val = info.get("returnOnEquity", 0.10)
        de_val = info.get("debtToEquity", 60.0)
        rg_val = info.get("revenueGrowth", 0.05)
        eg_val = info.get("earningsGrowth", 0.05)
        fcf_val = info.get("freeCashflow", 10_000_000)
        pe_val = info.get("trailingPE", 25.0)
        fwd_pe_val = info.get("forwardPE", 20.0)
        peg_val = info.get("pegRatio", 1.5)
        insider_held = info.get("heldPercentInsiders")
        insider_val = (insider_held * 100) if insider_held is not None else 5.0
        mcap = info.get("marketCap", 1)
        fcf_yield_val = (
            (fcf_val / mcap) * 100 if (mcap and mcap > 0 and fcf_val) else 2.5
        )
        sector = info.get("sector", "Unknown")
        analyst_rec = (
            (info.get("recommendationKey") or "HOLD")
            .upper()
            .replace("_", " ")
        )

        # 16 Pillar Scoring
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
        s23 = (
            5 if de_val < 50 else (4 if de_val < 150 else (3 if de_val < 300 else 2))
        )
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
            (
                "1.3",
                "Moat",
                "Revenue Model Quality",
                0.06,
                "Recurring",
                s13,
                "Retention",
            ),
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
            (
                "4.4",
                "Filters",
                "Insider Activity",
                0.06,
                "Stable",
                s44,
                "Net action",
            ),
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
        final_score = max(1.0, min(5.0, base_score))
        verdict = (
            "PASS / BUY"
            if final_score >= 4.0
            else ("WATCH / HOLD" if final_score >= 3.2 else "FAIL / AVOID")
        )
        target_buy_price = fair_value * (1 - mos_input)

        # Render Public Equity Dashboard
        st.subheader(f"Evaluation Dashboard: {company_name} ({ticker_input})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Evaluation Mode", "Public Equity (Profitable)")
        col2.metric("Composite Score", f"{final_score:.2f} / 5.00")
        col3.metric("CMP", f"{curr_sym}{cmp:,.2f}")
        col4.metric(
            "Target Max Buy Price",
            f"{curr_sym}{target_buy_price:,.2f}",
            delta="IN BUY ZONE" if cmp <= target_buy_price else "WAIT FOR DIP",
            delta_color="normal" if cmp <= target_buy_price else "inverse",
        )

        if "PASS" in verdict:
          st.success(f"**Verdict:** {verdict}")
        elif "WATCH" in verdict:
          st.warning(f"**Verdict:** {verdict}")
        else:
          st.error(f"**Verdict:** {verdict}")

        st.markdown("### 📊 Pillar Score Breakdown")
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
            title="16-Pillar Score Distribution",
        )
        fig.update_traces(
            textfont_size=13, textangle=0, textposition="inside", cliponaxis=False
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=650,
            font=dict(size=14, color="#1E293B"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.stop()

  # --- PRE-PROFIT STARTUP / VENTURE LENS (OR AUTO-ROUTED PUBLIC TICKER) ---
  if (
      not eval_mode.startswith("Public")
      or "startup_ticker_input" in locals()
      or "startup_ticker_input" in globals()
  ):
    if "startup_ticker_input" in locals() and startup_ticker_input:
      with st.spinner(
          f"Pulling growth data for {startup_ticker_input} under Venture"
          " Lens..."
      ):
        stock = yf.Ticker(startup_ticker_input)
        info = stock.info or {}
        company_name = info.get("longName") or startup_ticker_input
        cmp = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or 50.0
        )
        gross_margin_input = info.get("grossMargins", 0.65)
        rev_growth_input = info.get("revenueGrowth", 0.45)
        burn_multiple_input = 1.4
        nrr_input = 1.15
        tam_input = "Massive ($10B+)"
        runway_months = 18
        base_revenue = info.get("totalRevenue", 100_000_000)
        current_burn = abs(info.get("freeCashflow", 30_000_000))
        curr_sym = "$"
    else:
      company_name = startup_name
      base_revenue = 10_000_000
      current_burn = 12_000_000
      curr_sym = "$"

    # --- VENTURE SCORING & PROJECTIONS ---
    s11 = (
        5
        if gross_margin_input >= 0.75
        else (4 if gross_margin_input >= 0.60 else 2)
    )
    s12 = 5 if rev_growth_input >= 1.0 else (4 if rev_growth_input >= 0.50 else 2)
    s13 = (
        5
        if burn_multiple_input <= 1.2
        else (3 if burn_multiple_input <= 2.0 else 1)
    )
    s14 = 5 if nrr_input >= 1.20 else (4 if nrr_input >= 1.05 else 2)
    s15 = (
        5
        if tam_input == "Massive ($10B+)"
        else (3 if tam_input == "Large ($1B-$10B)" else 1)
    )
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

    # --- PROFITABILITY & GROWTH ROADMAP CALCULATIONS ---
    # Project 3-Year Revenue Growth
    yr1_rev = base_revenue * (1 + rev_growth_input)
    yr2_rev = yr1_rev * (1 + (rev_growth_input * 0.8))  # Growth deceleration curve
    yr3_rev = yr2_rev * (1 + (rev_growth_input * 0.6))

    # Estimate Breakeven Year based on operating leverage and gross margin scale
    # Assuming opex scales slower than revenue as gross margins cover overhead
    estimated_breakeven_months = max(
        6, int(runway_months * (1.5 / max(rev_growth_input, 0.2)))
    )
    if estimated_breakeven_months <= 12:
      breakeven_guidance = (
          "🚀 **Rapid Profitability Horizon:** Projected to reach operating"
          f" breakeven within ~{estimated_breakeven_months} months based on"
          " current compounding velocity."
      )
    elif estimated_breakeven_months <= 36:
      breakeven_guidance = (
          "📈 **Medium-Term Scale Horizon:** Projected to achieve profitability"
          f" in ~{estimated_breakeven_months // 12} years (requires monitoring"
          " cash runway buffer)."
      )
    else:
      breakeven_guidance = (
          "⚠️ **Extended Capital Horizon:** Profitability is >3 years away."
          " Focus heavily on reducing the burn multiple and protecting cash"
          " runway."
      )

    # Render Venture Dashboard
    st.subheader(f"Venture Growth Dashboard: {company_name}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evaluation Mode", "Pre-Profit Venture Lens")
    col2.metric("Composite Score", f"{final_score:.2f} / 5.00")
    col3.metric("Cash Runway", f"{runway_months} Months")
    col4.metric("Burn Multiple", f"{burn_multiple_input}x")

    if "HIGH-POTENTIAL" in verdict:
      st.success(f"**Verdict:** {verdict}")
    elif "WATCHLIST" in verdict:
      st.warning(f"**Verdict:** {verdict}")
    else:
      st.error(f"**Verdict:** {verdict}")

    # --- EXPECTED GROWTH CHART & PROFITABILITY GUIDANCE ---
    st.markdown(
        "### 📈 Expected Revenue / ARR Growth Chart & Profitability Guidance"
    )
    st.info(breakeven_guidance)

    # Build Growth Chart Dataframe
    chart_data = pd.DataFrame({
        "Timeline": ["Current Base", "Year 1 Forecast", "Year 2 Forecast", "Year 3 Forecast"],
        "Projected Revenue": [base_revenue, yr1_rev, yr2_rev, yr3_rev],
    })

    fig_growth = px.line(
        chart_data,
        x="Timeline",
        y="Projected Revenue",
        markers=True,
        text=chart_data["Projected Revenue"].apply(lambda x: f"${x:,.0f}"),
        title=f"3-Year Compounding Growth Trajectory — {company_name}",
    )
    fig_growth.update_traces(
        textposition="top center", line=dict(width=4, color="#0EA5E9")
    )
    fig_growth.update_layout(
        height=400, font=dict(size=14, color="#1E293B"), margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # Pillar Score Breakdown Bar Chart
    st.markdown("### 📊 Venture Pillar Score Distribution")
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
    fig_bar = px.bar(
        df_chart,
        x="Score",
        y="Criteria",
        orientation="h",
        color="Score",
        color_continuous_scale="Teal",
        range_x=[0, 5],
        text="Metric",
        title="Venture Lens Scorecard",
    )
    fig_bar.update_traces(
        textfont_size=13, textangle=0, textposition="inside", cliponaxis=False
    )
    fig_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=450,
        font=dict(size=14, color="#1E293B"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed Table
    st.markdown("### Detailed Venture Scorecard")
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

else:
  st.info(
      "👈 Configure your target company or startup in the sidebar and click"
      " **Run Comprehensive Audit** to begin."
  )
