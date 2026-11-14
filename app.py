import streamlit as st
import pandas as pd
import yfinance as yf
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule
from datetime import datetime

st.set_page_config(page_title="Valuation & Audit Suite", page_icon="📈", layout="wide")

st.title("📈 Anti-Fragile Stock & Startup Evaluation Suite")
st.markdown("Generate institutional-grade 16-pillar scorecards, live market valuations, or startup unit-economics with automatic pre-profit detection.")

# Sidebar navigation
app_mode = st.sidebar.selectbox("Select Evaluation Model", ["Public Stock Evaluator", "Pre-Profit Startup Evaluation Mode"])

if app_mode == "Public Stock Evaluator":
    st.subheader("🏢 Public Listed Company Valuation Engine")
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AVGO, TSLA, NVDA):", value="AVGO").strip().upper()

    if st.button("Generate Stock Scorecard"):
        with st.spinner(f"Fetching live market data and computing valuation for {ticker}..."):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                
                if not info or len(info) < 5:
                    raise ValueError(f"Could not retrieve valid data for '{ticker}'. It may be invalid or delisted.")

                cmp = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 100.0
                company_name = info.get('longName') or info.get('shortName') or ticker
                sector = info.get('sector', 'Unknown')
                fair_value = info.get('targetMeanPrice') or round(cmp * 1.25, 2)
                analyst_rec = (info.get('recommendationKey') or 'HOLD').upper().replace('_', ' ')
                num_analysts = info.get('numberOfAnalystOpinions', 5)

                gm_val = info.get('grossMargins') or 0.35
                op_margin = info.get('operatingMargins') or 0.12
                roe_val = info.get('returnOnEquity') or 0.10
                de_val = info.get('debtToEquity') or 60.0
                rg_val = info.get('revenueGrowth') or 0.05
                eg_val = info.get('earningsGrowth') or 0.05

                fcf_val = info.get('freeCashflow') or 0
                net_inc = info.get('netIncomeToCommon') or 0
                eps_val = info.get('trailingEps') or 0

                pe_val = info.get('trailingPE') or 0.0
                fwd_pe_val = info.get('forwardPE') or 0.0
                peg_val = info.get('pegRatio') or 0.0

                insider_held = info.get('heldPercentInsiders')
                insider_val = (insider_held * 100) if insider_held is not None else 5.0

                mcap = info.get('marketCap', 1)
                fcf_yield_val = (fcf_val / mcap) * 100 if (mcap and mcap > 0 and fcf_val) else 0.0

                # --- PROFITABILITY CHECK ---
                is_profitable = net_inc > 0 and eps_val > 0

                if not is_profitable:
                    st.error(f"🚨 **PROFITABILITY ALERT: NOT YET PROFITABLE**")
                    st.warning(f"**{company_name} ({ticker})** has **not yet turned profitable** based on trailing twelve months (TTM) data.\n"
                               f"- **Net Income:** ${net_inc:,.0f}\n"
                               f"- **Trailing EPS:** ${eps_val:.2f}\n"
                               f"*Note: Traditional P/E multiples and earnings-based quality filters have been adjusted for a pre-profit growth profile.*")
                else:
                    st.success(f"✅ **{company_name} ({ticker})** is currently profitable (Net Income: ${net_inc:,.0f} | EPS: ${eps_val:.2f}).")

                # 16 Pillars Scoring Logic (with pre-profit safeguards)
                s11 = 5 if gm_val >= 0.50 else (4 if gm_val >= 0.35 else (3 if gm_val >= 0.20 else 2))
                s12 = 5 if rg_val >= 0.15 else (4 if rg_val >= 0.08 else (3 if rg_val >= 0.0 else 2))
                s13 = 4
                s21 = 5 if roe_val >= 0.20 else (4 if roe_val >= 0.12 else (3 if roe_val >= 0.05 else 2))
                s22 = s11
                s23 = 5 if de_val < 50 else (4 if de_val < 150 else (3 if de_val < 300 else 2))
                
                fcf_ratio = (fcf_val / net_inc) if (net_inc and net_inc > 0) else 0.0
                s24 = 3 if not is_profitable else (5 if fcf_ratio >= 1.0 else (4 if fcf_ratio >= 0.75 else 3))
                
                s31 = 5 if fcf_yield_val >= 7.0 else (4 if fcf_yield_val >= 4.0 else (3 if fcf_yield_val >= 2.0 else 2))
                s32 = 3 if not is_profitable else (5 if (0 < peg_val <= 1.2) else 3)
                s33 = 4
                s34 = 5 if insider_val >= 10.0 else (4 if insider_val >= 3.0 else 2)
                
                s41, s42 = s21, s12
                s43 = 5 if fcf_val > 0 else 2
                s44 = 4
                s45 = 5 if ("BUY" in analyst_rec or "OUTPERFORM" in analyst_rec) else 3

                rows = [
                    ("1.1", "Moat", "Competitive Advantage", 0.06, f"Gross Margin: {gm_val*100:.1f}%", s11, "Pricing power"),
                    ("1.2", "Moat", "Market Dynamics", 0.06, f"Rev Growth: {rg_val*100:.1f}%", s12, "TAM growth"),
                    ("1.3", "Moat", "Revenue Model Quality", 0.06, "Recurring / Sticky", s13, "Retention"),
                    ("2.1", "Quality", "ROE / Capital Returns", 0.07, f"ROE: {roe_val*100:.1f}%", s21, "Efficiency"),
                    ("2.2", "Quality", "Gross Margin Floor", 0.07, f"GM: {gm_val*100:.1f}%", s22, "Margin stability"),
                    ("2.3", "Quality", "Debt Leverage", 0.07, f"D/E: {de_val:.1f}%", s23, "Solvency"),
                    ("2.4", "Quality", "FCF Conversion", 0.07, f"Net Inc: {'Negative' if not is_profitable else f'{fcf_ratio*100:.1f}%'}", s24, "Cash quality"),
                    ("3.1", "Valuation", "FCF Yield", 0.07, f"Yield: {fcf_yield_val:.1f}%", s31, "Cash yield"),
                    ("3.2", "Valuation", "PEG & P/E Multiple", 0.06, f"{'Pre-Profit' if not is_profitable else f'PEG: {peg_val:.2f}x'}", s32, "Growth multiple"),
                    ("3.3", "Governance", "Management Alignment", 0.06, "Shareholder returns", s33, "Capital policy"),
                    ("3.4", "Governance", "Insider Ownership", 0.05, f"Insiders: {insider_val:.1f}%", s34, "Skin in game"),
                    ("4.1", "Filters", "Multi-Yr Return Trend", 0.06, f"ROE: {roe_val*100:.1f}%", s41, "Durability"),
                    ("4.2", "Filters", "Top-Line Compounding", 0.06, f"Rev Growth: {rg_val*100:.1f}%", s42, "Expansion"),
                    ("4.3", "Filters", "Positive Free Cash Flow", 0.06, f"FCF: ${fcf_val:,.0f}", s43, "Self-funding"),
                    ("4.4", "Filters", "Insider Net Action", 0.06, "Stable", s44, "Recent activity"),
                    ("4.5", "Consensus", "Analyst Rating", 0.06, f"{analyst_rec} (${fair_value:,.2f})", s45, f"{num_analysts} analysts"),
                ]

                base_score = sum(wt * sc for _, _, _, wt, _, sc, _ in rows)
                final_score = max(1.0, min(5.0, base_score))

                col1, col2, col3 = st.columns(3)
                col1.metric("Final Score", f"{final_score:.2f} / 5.00")
                col2.metric("Net Income Status", "🔴 Pre-Profit" if not is_profitable else "🟢 Profitable")
                col3.metric("CMP", f"${cmp:,.2f}")

                st.subheader("📋 16-Pillar Scorecard Breakdown")
                st.dataframe(
                    [{"Ref": r[0], "Category": r[1], "Pillar": r[2], "Live Metric": r[4], "Score": r[5], "Notes": r[6]} for r in rows],
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:
                st.error(f"Error evaluating ticker: {e}")

elif app_mode == "Pre-Profit Startup Evaluation Mode":
    st.subheader("🚀 Pre-Profit / Early-Stage Startup Unit-Economics Engine")
    st.info("Use this mode to evaluate early-stage businesses where profitability has not yet been achieved.")
    burn_rate = st.number_input("Monthly Burn Rate ($)", value=50000)
    cash_reserves = st.number_input("Total Cash Reserves ($)", value=600000)
    runway_months = cash_reserves / burn_rate if burn_rate > 0 else 0
    st.metric("Estimated Runway", f"{runway_months:.1f} Months")
    if runway_months < 12:
        st.error("⚠️ **Critical Runway Alert:** Runway is under 12 months. Immediate fundraising or cost reduction required.")
    else:
        st.success("✅ **Healthy Runway:** Runway exceeds 12 months.")
