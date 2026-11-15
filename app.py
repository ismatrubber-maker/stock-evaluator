"""
Anti-Fragile Stock & Startup Screener (Streamlit)
=================================================
Run locally:  streamlit run app.py
Deploy:       put app.py and requirements.txt in the root of a GitHub repo, then point
              Streamlit Community Cloud at app.py.

Two lenses
  1. Public Equity Screen - 15 independent pillars for profitable listed companies (US and India).
  2. Venture Lens         - 6 pillars plus a cash-runway / breakeven model for pre-profit companies.

Design rules
  * Nothing is invented. A metric that cannot be sourced shows "n/a", is left out of the score and
    the remaining weights are rescaled. Data coverage is shown next to every score.
  * Every underlying metric is used once (no double counting).
  * Safety gates cap the result however high the composite score is.
  * Banks, insurers and REITs are blocked by default; utilities and cyclicals get a caution.
  * Output is a screen result (Strong / Mixed / Weak), not a buy or sell call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Anti-Fragile Stock & Startup Screener",
    page_icon="🛡️",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION  (tune thresholds here)
# ══════════════════════════════════════════════════════════════════════════════
CACHE_TTL = 60 * 60            # seconds a Yahoo snapshot stays cached
MIN_COVERAGE_PUBLIC = 0.70     # share of pillar weight that must have data to issue a result
MIN_COVERAGE_VENTURE = 0.65
STRONG_MIN = 3.8               # composite >= this -> Strong
MIXED_MIN = 3.0                # composite >= this -> Mixed, otherwise Weak
ROUTE_GROWTH_MIN = 0.15        # loss-making and revenue growth >= this -> Venture lens
MIN_PE = 15.0                  # lowest exit P/E the PEG rule can produce (about the long-run market average).
                               # PEG logic breaks down at low growth (5% growth would imply 7.5x), so it is floored.
EPS_GROWTH_FADE = 0.85         # year-2 EPS growth = year-1 growth x this
PROJECTION_YEARS = 5
PROJECTION_GROWTH_CAP = 1.5    # starting growth used in the venture projection is capped at 150%
DEFAULT_FADE = 0.80
DEFAULT_COST_RATIO = 0.60

MODE_PUBLIC = "Public equity screen (profitable companies)"
MODE_VENTURE = "Pre-profit / venture lens"
SRC_MANUAL = "Manual inputs"
SRC_TICKER = "Public ticker (auto-fetch)"
LOSS_AUTO = "Auto: growth companies go to the venture lens"
LOSS_VENTURE = "Always use the venture lens"
LOSS_SCREEN = "Always run the quality screen (capped at Weak)"
LOSS_OPTIONS = [LOSS_AUTO, LOSS_VENTURE, LOSS_SCREEN]

GROWTH_BASES = {
    "Conservative: lower of NI / revenue CAGR": min,
    "Balanced: average": lambda xs: sum(xs) / len(xs),
    "Optimistic: higher": max,
}

TAM_OPTIONS = ["Massive (10B+)", "Large (1B to 10B)", "Niche (under 1B)"]
TAM_UNKNOWN = "Unknown (leave out of score)"
TAM_CHOICES = TAM_OPTIONS + [TAM_UNKNOWN]
TAM_SCORES = {TAM_OPTIONS[0]: 5, TAM_OPTIONS[1]: 3, TAM_OPTIONS[2]: 1}

LABELS = {"strong": "Strong", "mixed": "Mixed", "weak": "Weak", "insufficient": "Insufficient data"}

# Sector / industry rules. "exclude" blocks the screen unless the user ticks the override;
# "caution" runs it with a warning. Yahoo files lenders and payment networks under the same sector,
# so Financial Services is decided by industry rather than blocked wholesale.
LENDER_KEYWORDS = ("Bank", "Insurance", "Mortgage", "Credit Services", "Capital Markets", "Thrift", "Conglomerate")
LENDER_TEXT = (
    "Banks, insurers, NBFCs and other lenders run on deposits and leverage, so gross margin, debt/equity "
    "and free cash flow do not mean what this rubric assumes. Use a bank-specific lens (net interest "
    "margin, gross/net NPA, capital adequacy, ROA, price/book). Yahoo files payment networks such as Visa "
    "and Mastercard under the same industry even though they do not lend: tick the sidebar checkbox to "
    "run them anyway."
)
FIN_CAUTION_TEXT = (
    "Financial-sector business model: gross margin and debt/equity may not be meaningful, so some "
    "pillars can read low for structural reasons."
)
REAL_ESTATE_TEXT = (
    "Real-estate and REIT cash flow and leverage are structurally different (depreciation, mortgage "
    "debt, FFO), so this rubric would mislead."
)
CYCLICAL_TEXT = "Cyclical sector: trailing margins and returns may reflect a commodity peak or trough."
UTILITY_TEXT = (
    "Regulated utilities carry high debt and capex by design, which this rubric penalises. "
    "Read low scores in that light."
)


def sector_rule(sector, industry=""):
    """Return (rule, explanation) for sectors the rubric fits poorly, else None."""
    sector, industry = sector or "", industry or ""
    if sector == "Financial Services":
        unknown = industry in ("", "Unknown")
        brokers = "Insurance Brokers" in industry
        if unknown or (any(k in industry for k in LENDER_KEYWORDS) and not brokers):
            return "exclude", LENDER_TEXT
        return "caution", FIN_CAUTION_TEXT
    if sector == "Real Estate":
        return "exclude", REAL_ESTATE_TEXT
    if sector == "Utilities":
        return "caution", UTILITY_TEXT
    if sector in ("Energy", "Basic Materials"):
        return "caution", CYCLICAL_TEXT
    return None

DISCLAIMER = (
    "Educational screening tool, not investment advice or a research report. Data comes from Yahoo "
    "Finance through the unofficial yfinance library and can be delayed, incomplete or wrong, "
    "especially for Indian tickers. Thresholds are uncalibrated heuristics. Check company filings "
    "before acting, and speak to a licensed adviser where the rules of your jurisdiction require it."
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMALL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
class DataError(Exception):
    """A usable data snapshot could not be built (bad ticker, empty response, ...)."""


def num(x):
    """Return x as a finite float, else None (handles None, NaN, inf, 'Infinity', strings)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def first_num(d, *keys):
    for k in keys:
        v = num(d.get(k))
        if v is not None:
            return v
    return None


def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def pos(x):
    """Keep only strictly positive values (prices and market caps can never be zero or negative)."""
    return x if x is not None and x > 0 else None


def nz(x):
    """Treat an exact zero as missing. Yahoo reports 0 for margins it could not compute (for example
    when a company has no gross-profit line), and a real margin is never exactly zero."""
    return None if x is None or x == 0 else x


def fmt_pct(v, d=1):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def fmt_x(v, d=1):
    return "n/a" if v is None else f"{v:.{d}f}x"


def fmt_price(v, sym="$"):
    return "n/a" if v is None else f"{sym}{v:,.2f}"


def fmt_big(v, sym="$"):
    """Compact money: $1.20B, ₹12,345 Cr."""
    if v is None:
        return "n/a"
    sign, a = ("-" if v < 0 else ""), abs(v)
    if sym == "₹":
        return f"{sign}₹{a / 1e7:,.0f} Cr" if a >= 1e7 else f"{sign}₹{a:,.0f}"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{sign}{sym}{a / div:,.2f}{suffix}"
    return f"{sign}{sym}{a:,.2f}"


def fmt_mm(v, sym="$"):
    """Format a value that is stored in millions."""
    return "n/a" if v is None else fmt_big(v * 1e6, sym)


CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CNY": "¥", "CAD": "C$", "AUD": "A$", "HKD": "HK$", "SGD": "S$",
}


def currency_symbol(code, ticker=""):
    if code in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[code]
    if not code:
        return "₹" if ticker.endswith((".NS", ".BO")) else "$"
    return f"{code} "


def esc(text):
    """Streamlit renders $...$ as LaTeX; escape dollar signs in markdown text."""
    return text.replace("$", "\\$")


def say(kind, text):
    """Show an info / success / warning / error box with dollar signs escaped."""
    getattr(st, kind)(esc(text))


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA LAYER  (yfinance -> plain numbers; everything below is pure and testable)
# ══════════════════════════════════════════════════════════════════════════════
def stmt_series(df, *labels):
    """Oldest-to-newest [(period_end, value)] for the first row label found in a yfinance statement."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    for label in labels:
        if label not in df.index:
            continue
        row = df.loc[label]
        if isinstance(row, pd.DataFrame):  # duplicated row labels
            row = row.iloc[0]
        row = pd.to_numeric(row, errors="coerce").dropna()
        pts = [(idx, float(v)) for idx, v in row.items() if math.isfinite(v)]
        try:
            pts.sort(key=lambda p: p[0])
        except TypeError:
            pass
        if pts:
            return pts
    return []


def latest(df, *labels):
    pts = stmt_series(df, *labels)
    return pts[-1][1] if pts else None


def margin_series(num_pts, den_pts):
    den = dict(den_pts)
    return [(d, n / den[d]) for d, n in num_pts if d in den and den[d] > 0]


def cagr(points, min_points=3):
    """Compound annual growth across (date, value) points -> (rate, years); (None, 0) if not computable."""
    if len(points) < min_points:
        return None, 0
    (d0, v0), (d1, v1) = points[0], points[-1]
    if v0 <= 0 or v1 <= 0:
        return None, 0
    try:
        years = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.25
    except Exception:
        years = len(points) - 1
    if years < 1:
        return None, 0
    return (v1 / v0) ** (1 / years) - 1, int(round(years))


def fcf_series(cf):
    pts = stmt_series(cf, "Free Cash Flow")
    if pts:
        return pts
    ocf = dict(stmt_series(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"))
    capex = dict(stmt_series(cf, "Capital Expenditure"))
    return [(d, ocf[d] - abs(capex[d])) for d in sorted(ocf) if d in capex]


def avg_return_on_capital(inc, bal, n_years=3):
    """Average ROIC = EBIT x (1 - tax) / invested capital over the latest n_years."""
    ebit = dict(stmt_series(inc, "EBIT", "Operating Income"))
    tax_calc = dict(stmt_series(inc, "Tax Rate For Calcs"))
    pretax = dict(stmt_series(inc, "Pretax Income"))
    tax = dict(stmt_series(inc, "Tax Provision"))
    inv = dict(stmt_series(bal, "Invested Capital"))
    debt = dict(stmt_series(bal, "Total Debt"))
    eq = dict(stmt_series(bal, "Stockholders Equity", "Common Stock Equity"))
    cash = dict(stmt_series(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"))
    vals = []
    for d in sorted(ebit):
        ic = inv.get(d)
        if ic is None and d in debt and d in eq:
            ic = debt[d] + eq[d] - cash.get(d, 0.0)
        if ic is None or ic <= 0:
            continue
        rate = tax_calc.get(d)
        if rate is None and pretax.get(d, 0) > 0 and d in tax:
            rate = tax[d] / pretax[d]
        rate = 0.25 if rate is None else min(max(rate, 0.0), 0.40)
        vals.append(ebit[d] * (1 - rate) / ic)
    vals = vals[-n_years:]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def avg_roe(inc, bal, n_years=3):
    ni = dict(stmt_series(inc, "Net Income", "Net Income Common Stockholders"))
    eq = dict(stmt_series(bal, "Stockholders Equity", "Common Stock Equity"))
    vals = [ni[d] / eq[d] for d in sorted(ni) if d in eq and eq[d] > 0][-n_years:]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def share_trend(points):
    """Annual change in share count -> (rate, note). Rejects series that look split-distorted."""
    if len(points) < 3:
        return None, "Fewer than 3 years of share-count data."
    vals = [v for _, v in points]
    for a, b in zip(vals, vals[1:]):
        if a <= 0 or abs(b / a - 1) > 0.35:
            return None, "Share count jumps by more than 35% in a year (likely a split), so it is excluded."
    rate, _ = cagr(points)
    return rate, ""


@st.cache_data(ttl=CACHE_TTL, max_entries=64, show_spinner=False)
def fetch_snapshot(ticker: str) -> dict:
    """Pull info and the three annual statements from Yahoo. Raises (and so is not cached) on failure."""
    tk = yf.Ticker(ticker)
    info = tk.info
    if not isinstance(info, dict) or not info:
        raise DataError(f"Yahoo returned no data for '{ticker}'. Check the symbol and try again.")
    price = first_num(info, "currentPrice", "regularMarketPrice", "previousClose")
    if price is None and not (info.get("longName") or info.get("shortName")):
        raise DataError(
            f"No market data found for '{ticker}'. Check the symbol: Indian stocks need .NS (NSE) or .BO (BSE), "
            "for example RELIANCE.NS."
        )
    if len(info) < 8:  # a real equity has 100+ fields; a handful means Yahoo answered badly (often rate limiting)
        raise DataError(
            f"Yahoo returned only partial data for '{ticker}' ({len(info)} fields), which often happens when it is "
            "rate-limiting a shared server. Wait a minute and click Run audit again."
        )
    frames = {}
    for key, attr in (("income", "financials"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
        try:
            df = getattr(tk, attr)
        except Exception:  # statements are optional; missing ones just reduce data coverage
            df = None
        frames[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    return {
        "ticker": ticker,
        "info": dict(info),
        "fetched_at": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        **frames,
    }


def extract_public_metrics(snap):
    """Turn a snapshot into the plain metrics the public screen needs. Missing -> None, never a default."""
    info, inc, bal, cf = snap["info"], snap["income"], snap["balance"], snap["cashflow"]
    t = snap["ticker"]
    cur, fcur = info.get("currency"), info.get("financialCurrency")
    m = {
        "ticker": t,
        "fetched_at": snap["fetched_at"],
        "name": info.get("longName") or info.get("shortName") or t,
        "sector": info.get("sector") or "Unknown",
        "industry": info.get("industry") or "Unknown",
        "currency": cur,
        "fin_currency": fcur,
        "sym": currency_symbol(cur, t),
        "fin_sym": currency_symbol(fcur or cur, t),
        "currency_mismatch": bool(cur and fcur and cur != fcur),
        "price": pos(first_num(info, "currentPrice", "regularMarketPrice", "previousClose")),
        "market_cap": pos(first_num(info, "marketCap")),
        "pb": first_num(info, "priceToBook"),
        "roe_latest": first_num(info, "returnOnEquity"),
    }

    # Income statement ---------------------------------------------------------
    rev_pts = stmt_series(inc, "Total Revenue", "Operating Revenue")
    ni_pts = stmt_series(inc, "Net Income", "Net Income Common Stockholders")
    gp_pts = stmt_series(inc, "Gross Profit")
    ebit_pts = stmt_series(inc, "EBIT", "Operating Income")
    int_pts = stmt_series(inc, "Interest Expense", "Interest Expense Non Operating")
    m["rev_ttm"] = coalesce(first_num(info, "totalRevenue"), rev_pts[-1][1] if rev_pts else None)
    m["ni_ttm"] = coalesce(first_num(info, "netIncomeToCommon"), ni_pts[-1][1] if ni_pts else None)
    m["rev_growth_yoy"] = first_num(info, "revenueGrowth")
    m["earn_growth"] = first_num(info, "earningsGrowth")

    gm_pts = margin_series(gp_pts, rev_pts)
    m["gross_margin"] = coalesce(nz(first_num(info, "grossMargins")), gm_pts[-1][1] if gm_pts else None)
    m["gm_trend_pp"] = (gm_pts[-1][1] - gm_pts[0][1]) * 100 if len(gm_pts) >= 3 else None
    om_pts = margin_series(ebit_pts, rev_pts)
    m["op_margin"] = coalesce(nz(first_num(info, "operatingMargins")), om_pts[-1][1] if om_pts else None)
    m["rev_cagr"], m["rev_years"] = cagr(rev_pts)
    m["ni_cagr"], m["ni_years"] = cagr(ni_pts)

    # Balance sheet and leverage -------------------------------------------------
    eq_pts = stmt_series(bal, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
    m["equity"] = eq_pts[-1][1] if eq_pts else None
    m["total_debt"] = coalesce(first_num(info, "totalDebt"), latest(bal, "Total Debt"))
    m["cash"] = coalesce(
        first_num(info, "totalCash"),
        latest(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"),
    )
    de = first_num(info, "debtToEquity")  # Yahoo reports this as a percentage
    if de is None and m["total_debt"] is not None and m["equity"] and m["equity"] > 0:
        de = m["total_debt"] / m["equity"] * 100
    m["neg_equity"] = bool((m["equity"] is not None and m["equity"] <= 0) or (de is not None and de < 0))
    m["de"] = None if m["neg_equity"] else de

    ebit_latest = ebit_pts[-1][1] if ebit_pts else None
    int_latest = abs(int_pts[-1][1]) if int_pts else None
    m["no_debt"] = False
    if ebit_latest is not None and int_latest:
        m["int_cov"] = ebit_latest / int_latest
    elif m["total_debt"] is not None and m["total_debt"] <= 0:
        m["int_cov"], m["no_debt"] = 99.0, True
    else:
        m["int_cov"] = None

    ebitda = first_num(info, "ebitda")
    td, cash = m["total_debt"], m["cash"]
    m["nd_ebitda"] = (td - cash) / ebitda if (td is not None and cash is not None and ebitda and ebitda > 0) else None

    # Returns on capital ---------------------------------------------------------
    roic, n = avg_return_on_capital(inc, bal)
    if roic is not None:
        m["roc"], m["roc_label"] = roic, f"ROIC (avg {n}y)"
    else:
        roe, n = avg_roe(inc, bal)
        if roe is not None:
            m["roc"], m["roc_label"] = roe, f"ROE (avg {n}y)"
        elif m["roe_latest"] is not None and not m["neg_equity"]:
            m["roc"], m["roc_label"] = m["roe_latest"], "ROE (latest)"
        else:
            m["roc"], m["roc_label"] = None, "ROIC"

    # Cash flow ------------------------------------------------------------------
    fcf_pts = fcf_series(cf)[-4:]
    m["fcf_ttm"] = coalesce(first_num(info, "freeCashflow"), fcf_pts[-1][1] if fcf_pts else None)
    m["fcf_years"] = len(fcf_pts)
    m["fcf_neg_years"] = sum(1 for _, v in fcf_pts if v < 0)
    ni, fcf, mc = m["ni_ttm"], m["fcf_ttm"], m["market_cap"]
    m["fcf_conv"] = fcf / ni if (fcf is not None and ni is not None and ni > 0) else None
    # FCF is in the reporting currency and market cap in the trading currency: only compare if they match.
    m["fcf_yield"] = fcf / mc if (fcf is not None and mc and mc > 0 and not m["currency_mismatch"]) else None
    sbc = latest(cf, "Stock Based Compensation")
    rev_latest = rev_pts[-1][1] if rev_pts else None
    m["sbc_pct"] = sbc / rev_latest if (sbc is not None and rev_latest and rev_latest > 0) else None

    # Valuation ------------------------------------------------------------------
    m["eps_ttm"] = first_num(info, "trailingEps")
    pe = first_num(info, "trailingPE")
    if pe is None and m["price"] and m["eps_ttm"] and m["eps_ttm"] > 0:
        pe = m["price"] / m["eps_ttm"]
    m["pe_ttm"] = pe if (pe is not None and pe > 0) else None
    peg = next((v for v in (num(info.get("pegRatio")), num(info.get("trailingPegRatio"))) if v is not None and v > 0), None)
    m["peg_source"] = "Yahoo"
    if peg is None and m["pe_ttm"] and m["ni_cagr"] and m["ni_cagr"] >= 0.03:
        peg, m["peg_source"] = m["pe_ttm"] / (m["ni_cagr"] * 100), "P/E ÷ NI CAGR"
    m["peg"] = peg

    # Governance and sentiment ------------------------------------------------------
    ins = first_num(info, "heldPercentInsiders")
    m["insider_pct"] = ins * 100 if ins is not None else None
    m["share_cagr"], m["share_note"] = share_trend(stmt_series(bal, "Ordinary Shares Number", "Share Issued"))
    mean, n_an = first_num(info, "recommendationMean"), first_num(info, "numberOfAnalystOpinions")
    m["analyst_mean"] = mean if (mean is not None and n_an is not None and n_an >= 3) else None
    m["analyst_n"] = n_an
    m["analyst_target"] = first_num(info, "targetMeanPrice")  # shown for reference only, never used for valuation
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORING MODEL (shared by both lenses)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Pillar:
    ref: str
    group: str
    name: str
    weight: float             # percent; only relative weights matter
    value: Optional[float]    # raw metric, None = unavailable
    display: str              # formatted metric, "n/a" when unavailable
    score: Optional[int]      # 1 (poor) .. 5 (excellent); None = left out of the score
    note: str = ""


def tier(value, cuts, higher=True):
    """Map a metric to 5..1 using four cut-offs ordered from best to worst. None stays None."""
    if value is None:
        return None
    for score, cut in zip((5, 4, 3, 2), cuts):
        if (value >= cut) if higher else (value <= cut):
            return score
    return 1


def summarize(pillars):
    """Weighted composite over pillars that have data -> (score, coverage, group_scores)."""
    total_w = sum(p.weight for p in pillars)
    have = [p for p in pillars if p.score is not None]
    w_have = sum(p.weight for p in have)
    coverage = w_have / total_w if total_w else 0.0
    score = sum(p.weight * p.score for p in have) / w_have if w_have else None
    groups = {}
    for g in dict.fromkeys(p.group for p in pillars):
        gp = [p for p in have if p.group == g]
        gw = sum(p.weight for p in gp)
        groups[g] = sum(p.weight * p.score for p in gp) / gw if gw else None
    return score, coverage, groups


def decide(score, coverage, min_cov, hard, soft):
    """Composite + coverage + safety gates -> (kind, label, reason)."""
    if score is None or coverage < min_cov:
        return (
            "insufficient",
            LABELS["insufficient"],
            f"Only {coverage:.0%} of the pillar weight has data (minimum {min_cov:.0%}), so no result is issued. "
            "Try another ticker or fill in the missing inputs.",
        )
    kind = "strong" if score >= STRONG_MIN else "mixed" if score >= MIXED_MIN else "weak"
    reason = f"Composite score {score:.2f} out of 5."
    if hard:
        reason += " Capped at Weak by a hard safety gate." if kind != "weak" else " A hard safety gate also triggered."
        kind = "weak"
    elif soft and kind == "strong":
        reason += " Capped at Mixed by a safety gate."
        kind = "mixed"
    return kind, LABELS[kind], reason


# ══════════════════════════════════════════════════════════════════════════════
# 5. PUBLIC EQUITY SCREEN
# ══════════════════════════════════════════════════════════════════════════════
def build_public_pillars(m):
    """15 pillars, each measuring a different thing. Weights add to 100."""
    out = []

    def add(ref, group, name, weight, value, fmt, cuts, higher, note):
        out.append(Pillar(ref, group, name, weight, value, fmt(value) if value is not None else "n/a",
                          tier(value, cuts, higher), note))

    g1, g2, g3, g4 = "Moat & growth", "Returns & balance sheet", "Valuation", "Governance & capital allocation"
    pct = lambda v: f"{v * 100:.1f}%"

    add("1.1", g1, "Gross margin level", 7, m["gross_margin"], pct, (0.50, 0.35, 0.20, 0.10), True,
        "Pricing power. Absolute thresholds, not sector-adjusted.")
    add("1.2", g1, "Gross margin trend", 5, m["gm_trend_pp"], lambda v: f"{v:+.1f} pp", (2, -1, -4, -8), True,
        "Change across the available annual filings. Erosion is an early moat warning.")
    add("1.3", g1, "Operating margin", 6, m["op_margin"], pct, (0.25, 0.15, 0.08, 0.03), True,
        "Cost discipline after overheads. Not sector-adjusted.")
    add("1.4", g1, "Revenue CAGR", 7, m["rev_cagr"], lambda v: f"{v * 100:.1f}% ({m['rev_years']}y)",
        (0.15, 0.08, 0.03, 0.0), True, "Compound growth from annual filings (needs 3+ years).")
    add("1.5", g1, "Net income CAGR", 6, m["ni_cagr"], lambda v: f"{v * 100:.1f}% ({m['ni_years']}y)",
        (0.15, 0.08, 0.03, 0.0), True, "Needs positive net income at both ends of the period.")

    add("2.1", g2, "Return on capital", 10, m["roc"], lambda v: f"{m['roc_label']}: {v * 100:.1f}%",
        (0.20, 0.15, 0.10, 0.05), True, "ROIC preferred. Falls back to average ROE, which leverage can inflate.")
    add("2.2", g2, "Debt / equity", 7, m["de"], lambda v: f"{v:.0f}%", (30, 75, 150, 250), False,
        "Yahoo reports a percentage and includes leases. n/a when book equity is negative.")
    add("2.3", g2, "Interest coverage", 8, m["int_cov"],
        lambda v: "No debt" if m["no_debt"] else (">50x" if v > 50 else f"{v:.1f}x"),
        (10, 6, 3, 1.5), True, "EBIT divided by interest expense (latest annual filing).")
    add("2.4", g2, "FCF conversion", 7, m["fcf_conv"], lambda v: f"{v * 100:.0f}% of net income",
        (1.0, 0.8, 0.6, 0.3), True, "Free cash flow / net income: are the earnings backed by cash?")
    yrs, neg = m["fcf_years"], m["fcf_neg_years"]
    add("2.5", g2, "FCF consistency", 7, (yrs - neg) / yrs if yrs >= 3 else None,
        lambda v: f"{yrs - neg} of {yrs} years positive", (1.0, 0.75, 0.5, 0.25), True,
        "Share of recent annual filings with positive free cash flow (needs 3+ years).")

    add("3.1", g3, "FCF yield", 8, m["fcf_yield"], pct, (0.06, 0.04, 0.025, 0.01), True,
        "Trailing FCF / market cap. Skipped if reporting and trading currencies differ.")
    add("3.2", g3, "PEG", 7, m["peg"], lambda v: f"{v:.2f}x ({m['peg_source']})", (1.0, 1.5, 2.0, 3.0), False,
        "P/E relative to growth. n/a when growth or P/E is not meaningful.")

    add("4.1", g4, "Insider ownership", 5, m["insider_pct"], lambda v: f"{v:.1f}%", (10, 3, 1, 0.25), True,
        "Skin in the game. Low is normal for professionally managed large caps; promoter pledging is not captured.")
    add("4.2", g4, "Share count trend", 7, m["share_cagr"], lambda v: f"{v * 100:+.1f}% a year",
        (-0.015, 0.005, 0.02, 0.04), False, m["share_note"] or "Negative = net buybacks, positive = dilution.")
    add("4.3", g4, "Analyst consensus", 3, m["analyst_mean"], lambda v: f"{v:.1f} (1 = strong buy, 5 = sell)",
        (1.8, 2.4, 3.0, 3.5), False, "Sentiment only, so low weight. Needs 3+ analysts.")
    return out


def public_gates(m):
    """Returns (hard_gates, soft_gates, red_flags), each a list of (title, detail)."""
    hard, soft, flags = [], [], []

    if m["ni_ttm"] is not None and m["ni_ttm"] <= 0:
        hard.append(("Loss-making", "Trailing net income is zero or negative, so this is not a profitable compounder."))
    ic = m["int_cov"]
    if ic is not None and ic < 1.5:
        hard.append(("Interest coverage below 1.5x", f"EBIT covers interest only {ic:.1f}x."))
    elif ic is not None and ic < 3:
        soft.append(("Interest coverage below 3x", f"EBIT covers interest only {ic:.1f}x."))
    if m["fcf_years"] >= 3 and m["fcf_neg_years"] >= 3:
        hard.append(("Persistent negative free cash flow",
                     f"FCF was negative in {m['fcf_neg_years']} of the last {m['fcf_years']} annual filings."))
    if m["de"] is not None and m["de"] > 150:
        soft.append(("Debt / equity above 150%", f"Debt is {m['de']:.0f}% of book equity."))
    if m["nd_ebitda"] is not None and m["nd_ebitda"] > 4:
        soft.append(("Net debt above 4x EBITDA", f"Net debt is {m['nd_ebitda']:.1f}x EBITDA."))
    if m["fcf_ttm"] is not None and m["fcf_ttm"] < 0:
        soft.append(("Negative trailing free cash flow", "The business is not currently self-funding."))
    if m["share_cagr"] is not None and m["share_cagr"] > 0.03:
        soft.append(("Heavy share dilution", f"Share count is growing {m['share_cagr'] * 100:.1f}% a year."))
    if m["rev_cagr"] is not None and m["rev_cagr"] < 0:
        soft.append(("Shrinking revenue", f"Revenue CAGR is {m['rev_cagr'] * 100:.1f}%."))

    if m["peg"] is not None and m["peg"] > 2.5:
        flags.append(("PEG above 2.5x", f"PEG is {m['peg']:.2f}x: the price already assumes strong growth."))
    if m["roc"] is not None and m["roc"] < 0.08:
        flags.append(("Low capital efficiency",
                      f"{m['roc_label']} is {m['roc'] * 100:.1f}%, below a typical 8-10% cost of capital."))
    if m["fcf_conv"] is not None and m["fcf_conv"] < 0.5:
        flags.append(("Weak cash conversion", f"FCF is only {m['fcf_conv'] * 100:.0f}% of net income."))
    if m["sbc_pct"] is not None and m["sbc_pct"] > 0.10:
        flags.append(("Heavy stock-based compensation",
                      f"SBC is {m['sbc_pct'] * 100:.0f}% of revenue, a real cost that FCF hides."))
    if m["op_margin"] is not None and m["op_margin"] < 0.05:
        flags.append(("Thin operating margin", f"Operating margin is {m['op_margin'] * 100:.1f}%."))
    if m["neg_equity"]:
        flags.append(("Negative book equity", "D/E and ROE are not meaningful. Check net debt / EBITDA instead."))
    return hard, soft, flags


def forward_model(m, cfg):
    """Illustrative 2-year EPS / price path -> (result dict or None, reason if None).

    Growth = conservative multi-year growth, capped. Exit P/E = lower of today's P/E and the
    PEG-implied P/E, so the multiple can stay flat or shrink but never expands. Fair value = year-2
    price discounted at the required return. The entry price adds a margin of safety. Analyst targets
    are never used.
    """
    if m["currency_mismatch"]:
        return None, ("Prices are quoted in one currency and financials reported in another (typical for ADRs), "
                      "so per-share forecasts are skipped to avoid mixing currencies.")
    eps0, price = m["eps_ttm"], m["price"]
    if eps0 is None or eps0 <= 0 or price is None or price <= 0:
        return None, "Trailing EPS or price is unavailable (or EPS is not positive), so no forward valuation is produced."
    hist = [x for x in (m["ni_cagr"], m["rev_cagr"]) if x is not None]
    if hist:
        g_raw = GROWTH_BASES[cfg["growth_basis"]](hist)
    elif m["earn_growth"] is not None:
        g_raw = m["earn_growth"]
    else:
        return None, "No growth history is available, so no forward valuation is produced."
    g = min(max(g_raw, 0.0), cfg["max_growth"])
    pe_peg = min(cfg["max_pe"], max(MIN_PE, cfg["target_peg"] * g * 100))
    mult = min(m["pe_ttm"], pe_peg) if m["pe_ttm"] else pe_peg
    eps1 = eps0 * (1 + g)
    eps2 = eps1 * (1 + g * EPS_GROWTH_FADE)
    p1, p2 = eps1 * mult, eps2 * mult
    fair = p2 / (1 + cfg["req_return"]) ** 2
    entry = fair * (1 - cfg["mos"])
    ni0 = m["ni_ttm"]
    return {
        "g_raw": g_raw, "g": g, "pe_peg": pe_peg, "mult": mult,
        "eps": [eps0, eps1, eps2],
        "ni": [ni0, ni0 * eps1 / eps0 if ni0 is not None else None, ni0 * eps2 / eps0 if ni0 is not None else None],
        "prices": [price, p1, p2],
        "fair": fair, "entry": entry,
        "implied": (p2 / price) ** 0.5 - 1,
    }, ""


# ══════════════════════════════════════════════════════════════════════════════
# 6. VENTURE LENS
# ══════════════════════════════════════════════════════════════════════════════
def venture_inputs_from_snapshot(snap, nrr=None, tam=None):
    """Public growth company -> venture inputs. Values are in millions. NRR and TAM are never guessed."""
    info, inc, cf = snap["info"], snap["income"], snap["cashflow"]
    t = snap["ticker"]
    sym = currency_symbol(info.get("financialCurrency") or info.get("currency"), t)
    rev_pts = stmt_series(inc, "Total Revenue", "Operating Revenue")
    rev = coalesce(first_num(info, "totalRevenue"), rev_pts[-1][1] if rev_pts else None)
    growth, growth_src = first_num(info, "revenueGrowth"), "Latest-quarter YoY from Yahoo (not ARR)"
    if growth is None and len(rev_pts) >= 2 and rev_pts[-2][1] > 0:
        growth, growth_src = rev_pts[-1][1] / rev_pts[-2][1] - 1, "Last two annual filings"
    gm_pts = margin_series(stmt_series(inc, "Gross Profit"), rev_pts)
    gm = coalesce(nz(first_num(info, "grossMargins")), gm_pts[-1][1] if gm_pts else None)
    fcf_pts = fcf_series(cf)
    fcf = coalesce(first_num(info, "freeCashflow"), fcf_pts[-1][1] if fcf_pts else None)
    cash, debt = first_num(info, "totalCash"), first_num(info, "totalDebt")
    mm = lambda v: v / 1e6 if v is not None else None
    return {
        "name": info.get("longName") or info.get("shortName") or t,
        "ticker": t, "sym": sym, "source": "ticker", "fetched_at": snap["fetched_at"],
        "revenue": mm(rev), "growth": growth, "growth_src": growth_src, "gross_margin": gm,
        "nrr": nrr, "tam": tam,
        "cash": mm(cash), "debt": mm(debt),
        "burn": mm(-fcf) if fcf is not None else None,  # positive = burning cash
    }


def venture_derive(vi):
    """Burn multiple, runway and Rule of 40 from raw inputs, so the inputs can never contradict each other."""
    rev, g, burn, cash = vi["revenue"], vi["growth"], vi["burn"], vi["cash"]
    net_new = rev * g / (1 + g) if (rev and g is not None and g > -1) else None
    if burn is None:
        bm = None
    elif burn <= 0:
        bm = 0.0                       # self-funding
    elif net_new is None:
        bm = None
    elif net_new <= 0:
        bm = math.inf                  # burning cash without adding revenue
    else:
        bm = burn / net_new
    if burn is not None and burn <= 0:
        runway = math.inf              # self-funding
    elif burn is None or cash is None:
        runway = None
    else:
        runway = cash / (burn / 12)
    rule40 = (g - burn / rev) * 100 if (rev and g is not None and burn is not None) else None
    return {
        "net_new": net_new, "burn_multiple": bm, "runway": runway, "rule40": rule40,
        "burning": burn is not None and burn > 0,
    }


def build_venture_pillars(vi, d):
    out = []

    def add(ref, group, name, weight, value, fmt, cuts, higher, note):
        out.append(Pillar(ref, group, name, weight, value, fmt(value) if value is not None else "n/a",
                          tier(value, cuts, higher), note))

    def fmt_bm(v):
        return "Self-funding" if v == 0 else ("No net new revenue" if math.isinf(v) else f"{v:.2f}x")

    def fmt_runway(v):
        return "Self-funding" if math.isinf(v) else f"{v:.0f} months"

    add("V1", "Unit economics", "Gross margin", 15, vi["gross_margin"], lambda v: f"{v * 100:.1f}%",
        (0.75, 0.60, 0.45, 0.30), True, "Room to fund R&D and sales as the company scales.")
    add("V2", "Growth", "Revenue / ARR growth (YoY)", 20, vi["growth"], lambda v: f"{v * 100:.0f}%",
        (1.0, 0.5, 0.25, 0.10), True, vi.get("growth_src") or "")
    add("V3", "Efficiency", "Burn multiple", 20, d["burn_multiple"], fmt_bm, (1.0, 1.5, 2.0, 3.0), False,
        "Net burn / net new revenue. At or under 1x is excellent, over 3x is poor.")
    add("V4", "Retention", "Net revenue retention", 15, vi["nrr"], lambda v: f"{v * 100:.0f}%",
        (1.20, 1.10, 1.00, 0.90), True, "Not available from Yahoo. Enter it yourself if you know it.")
    tam_score = TAM_SCORES.get(vi["tam"])
    out.append(Pillar("V5", "Market", "Addressable market (TAM)", 10,
                      float(tam_score) if tam_score else None, vi["tam"] or "n/a", tam_score,
                      "Your own judgement; leave it unknown rather than guess."))
    add("V6", "Risk", "Cash runway", 20, d["runway"], fmt_runway, (24, 18, 12, 6), True,
        "Cash / monthly net burn. Ignores debt maturities and any planned raise.")
    return out


def venture_gates(vi, d):
    hard, soft, flags = [], [], []
    rw, bm = d["runway"], d["burn_multiple"]
    if rw is not None and rw < 12:
        hard.append(("Runway under 12 months", f"About {rw:.0f} months of cash at the current burn."))
    elif rw is not None and rw < 18:
        soft.append(("Runway under 18 months", f"About {rw:.0f} months of cash at the current burn."))
    if bm is not None and bm > 3:
        hard.append(("Burn multiple above 3x", "Cash is being converted into new revenue very inefficiently."))
    if vi["nrr"] is not None and vi["nrr"] < 1.0:
        soft.append(("Net revenue retention below 100%", "The existing customer base is shrinking."))
    if vi.get("debt") and vi["cash"] is not None and d["burning"] and vi["debt"] > vi["cash"]:
        soft.append(("Debt exceeds cash while burning", "Refinancing risk: debt is larger than the cash buffer."))

    if vi["gross_margin"] is not None and vi["gross_margin"] < 0.40:
        flags.append(("Low gross margin", f"At {vi['gross_margin'] * 100:.0f}% there is little room to fund opex as it scales."))
    if vi["growth"] is not None and vi["growth"] < 0.25 and d["burning"]:
        flags.append(("Burning cash at low growth", f"Growth of {vi['growth'] * 100:.0f}% may not justify the burn."))
    if vi["growth"] is not None and vi["growth"] > PROJECTION_GROWTH_CAP:
        flags.append(("Growth capped in the projection",
                      f"Only {PROJECTION_GROWTH_CAP:.0%} is used as starting growth; extreme rates do not persist."))
    if vi["source"] == "ticker":
        flags.append(("Burn is a free-cash-flow proxy",
                      "It includes capex, which can overstate operating burn for capital-intensive companies, "
                      "and excludes stock-based compensation."))
    return hard, soft, flags


def project_venture(vi, fade, cost_ratio):
    """Simple operating model -> (DataFrame by year, summary). All amounts in millions.

    Revenue grows at start_growth x fade^t. Gross profit = revenue x gross margin. The cash cost base
    (everything below gross profit, derived from today's burn) grows at cost_ratio x revenue growth,
    which is the operating leverage assumption. Net burn = cost base - gross profit.
    """
    rev0, gm, burn0, cash0 = vi["revenue"], vi["gross_margin"], vi["burn"], vi["cash"]
    g0 = min(vi["growth"], PROJECTION_GROWTH_CAP)
    gp = rev0 * gm
    cost = max(gp + burn0, 0.0)
    cash = cash0 if cash0 is not None else float("nan")
    rows = [{"Year": 0, "Revenue": rev0, "Gross profit": gp, "Cash costs": cost, "Net burn": burn0, "Cash": cash}]
    rev = rev0
    for t in range(1, PROJECTION_YEARS + 1):
        g_t = g0 * fade ** t
        rev *= 1 + g_t
        gp = rev * gm
        cost *= 1 + g_t * cost_ratio
        burn = cost - gp
        cash -= burn
        rows.append({"Year": t, "Revenue": rev, "Gross profit": gp, "Cash costs": cost, "Net burn": burn, "Cash": cash})
    df = pd.DataFrame(rows)

    burns = df["Net burn"].tolist()
    be_months, be_idx = None, None
    if burns[0] <= 0:
        be_months, be_idx = 0.0, 0
    else:
        for i in range(1, len(burns)):
            if burns[i] <= 0:
                frac = burns[i - 1] / (burns[i - 1] - burns[i])
                be_months, be_idx = ((i - 1) + frac) * 12, i
                break
    upto = burns[1:(be_idx + 1)] if be_idx is not None else burns[1:]
    cum_burn = sum(b for b in upto if b > 0)
    min_cash = df["Cash"].min()
    gap = None if (cash0 is None or math.isnan(min_cash)) else max(0.0, -min_cash)
    return df, {"breakeven_months": be_months, "cum_burn": cum_burn, "funding_gap": gap}


def breakeven_message(summary):
    be = summary["breakeven_months"]
    if be is None:
        return "warning", (f"Not projected to reach cash breakeven within {PROJECTION_YEARS} years at these "
                           "assumptions. Growth would have to hold up, or costs grow more slowly than assumed.")
    if be == 0:
        return "success", "Already at or above cash breakeven (net burn is zero or negative)."
    years = be / 12
    if be <= 24:
        return "success", f"Near-term breakeven: about {be:.0f} months ({years:.1f} years) in this illustrative model."
    if be <= 48:
        return "info", f"Medium-term breakeven: about {years:.1f} years in this illustrative model. Watch the cash buffer."
    return "warning", f"Extended horizon: breakeven is about {years:.1f} years out. The funding line below shows whether cash lasts that long."


def funding_message(summary, cash, sym):
    gap, cum = summary["funding_gap"], summary["cum_burn"]
    if gap is None:
        return "info", (f"Cumulative burn before breakeven is about {fmt_mm(cum, sym)}. Cash on hand is unknown, "
                        "so the funding gap cannot be computed.")
    if gap > 0:
        return "warning", (f"Projected funding gap of about {fmt_mm(gap, sym)}: cash runs out before breakeven, "
                           "which implies new equity (dilution) or debt.")
    return "success", f"Cash on hand ({fmt_mm(cash, sym)}) covers the projected burn of about {fmt_mm(cum, sym)} through breakeven."


# ══════════════════════════════════════════════════════════════════════════════
# 7. UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def pillar_dataframe(pillars):
    return pd.DataFrame([
        {
            "Ref": p.ref,
            "Group": p.group,
            "Pillar": p.name,
            "Weight": f"{p.weight:.0f}%",
            "Metric": p.display,
            "Score": str(p.score) if p.score is not None else "-",
            "Why it matters / caveat": p.note,
        }
        for p in pillars
    ])


def pillar_chart(pillars, title):
    scored = [p for p in pillars if p.score is not None]
    if not scored:
        return None
    df = pd.DataFrame({
        "Pillar": [f"{p.ref} {p.name}" for p in scored],
        "Score": [p.score for p in scored],
        "Metric": [p.display for p in scored],
    })
    fig = px.bar(df, x="Score", y="Pillar", orientation="h", color="Score", color_continuous_scale="Teal",
                 range_color=[1, 5], range_x=[0, 5], text="Metric", title=title)
    fig.update_traces(textposition="auto", cliponaxis=False)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(df["Pillar"].tolist())))
    fig.update_layout(height=max(320, 34 * len(df) + 120), coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def show_verdict(kind, label, reason):
    say({"strong": "success", "mixed": "warning", "weak": "error", "insufficient": "info"}[kind],
        f"**Screen result: {label}.** {reason}")


def show_findings(hard, soft, flags):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🚧 Safety gates")
        if not (hard or soft):
            say("success", "No safety gate triggered by the available data.")
        for title, detail in hard:
            say("error", f"**{title}.** {detail} Hard gate: caps the result at Weak.")
        for title, detail in soft:
            say("warning", f"**{title}.** {detail} Soft gate: caps the result at Mixed.")
    with c2:
        st.markdown("#### 🚩 Red flags")
        if not flags:
            say("success", "No red flag triggered by the available data.")
        for title, detail in flags:
            say("warning", f"**{title}.** {detail}")


def show_scorecard(pillars, filename, title):
    fig = pillar_chart(pillars, title)
    if fig is not None:
        st.plotly_chart(fig)
    missing = [f"{p.ref} {p.name}" for p in pillars if p.score is None]
    if missing:
        st.caption("Left out of the score because no usable data was found: " + ", ".join(missing) + ".")
    df = pillar_dataframe(pillars)
    st.dataframe(df, hide_index=True)
    st.download_button("Download scorecard (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")


def raw_inputs_table(m):
    s, fs = m["sym"], m["fin_sym"]
    items = [
        ("Price", fmt_price(m["price"], s)),
        ("Market cap", fmt_big(m["market_cap"], s)),
        ("Revenue (TTM)", fmt_big(m["rev_ttm"], fs)),
        ("Net income (TTM)", fmt_big(m["ni_ttm"], fs)),
        ("Free cash flow (TTM)", fmt_big(m["fcf_ttm"], fs)),
        ("Total debt", fmt_big(m["total_debt"], fs)),
        ("Cash and short-term investments", fmt_big(m["cash"], fs)),
        ("Book equity (latest annual)", fmt_big(m["equity"], fs)),
        ("Trailing EPS", fmt_price(m["eps_ttm"], s)),
        ("Trailing P/E", fmt_x(m["pe_ttm"])),
        (f"PEG ({m['peg_source']})", fmt_x(m["peg"], 2)),
        ("Revenue CAGR", "n/a" if m["rev_cagr"] is None else f"{fmt_pct(m['rev_cagr'])} over {m['rev_years']} years"),
        ("Net income CAGR", "n/a" if m["ni_cagr"] is None else f"{fmt_pct(m['ni_cagr'])} over {m['ni_years']} years"),
        ("Latest-quarter revenue growth (Yahoo)", fmt_pct(m["rev_growth_yoy"])),
        ("Net debt / EBITDA", fmt_x(m["nd_ebitda"])),
        ("Stock-based compensation / revenue", fmt_pct(m["sbc_pct"])),
        ("Annual filings with free cash flow", f"{m['fcf_years']} (negative in {m['fcf_neg_years']})"),
        ("Analyst consensus target (reference only)", fmt_price(m["analyst_target"], s)),
    ]
    return pd.DataFrame(items, columns=["Input", "Value"])


def show_forward(m, fwd, msg, cfg):
    if fwd is None:
        say("info", msg)
        return
    s, fs = m["sym"], m["fin_sym"]
    st.caption("Illustrative model, not a forecast. Every assumption is listed below and can be changed in the sidebar.")
    df = pd.DataFrame({
        "Period": ["Today (TTM)", "Year 1", "Year 2"],
        "EPS": [fmt_price(x, s) for x in fwd["eps"]],
        "Net profit": [fmt_big(x, fs) for x in fwd["ni"]],
        "P/E applied": [fmt_x(m["pe_ttm"]), fmt_x(fwd["mult"]), fmt_x(fwd["mult"])],
        "Price (today) / target": [fmt_price(x, s) for x in fwd["prices"]],
    })
    st.dataframe(df, hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fair value today (model)", fmt_price(fwd["fair"], s))
    c2.metric("Max entry price", fmt_price(fwd["entry"], s), help="Fair value less the margin of safety.")
    c3.metric("Implied annual return at today's price", f"{fwd['implied'] * 100:+.1f}%",
              help="Price return only (dividends excluded), if the year-2 target is reached.")
    c4.metric("Analyst consensus target", fmt_price(m["analyst_target"], s), help="Shown for reference only; not used.")
    st.caption(esc(
        f"Growth used: {fwd['g'] * 100:.1f}% (basis: {cfg['growth_basis'].split(':')[0].lower()}; raw {fwd['g_raw'] * 100:.1f}%, "
        f"floored at 0 and capped at {cfg['max_growth'] * 100:.0f}%). Year-2 growth fades to {EPS_GROWTH_FADE:.0%} of that. "
        f"Exit P/E = lower of today's P/E and the PEG-implied P/E ({cfg['target_peg']:.1f} x growth, between {MIN_PE:.0f} "
        f"and {cfg['max_pe']:.0f}), so the multiple can stay flat or shrink but never expands. Fair value discounts "
        f"the year-2 price at {cfg['req_return'] * 100:.0f}% a year; the margin of safety is {cfg['mos'] * 100:.0f}%. "
        "Because growth is capped and the multiple cannot expand, high-multiple growth stocks will usually sit above "
        "the entry price. Read it as what you would have to believe, not as a price target."
    ))
    bars = pd.DataFrame({
        "Item": ["Price today", "Year-1 target", "Year-2 target", "Fair value today", "Max entry price"],
        "Value": [fwd["prices"][0], fwd["prices"][1], fwd["prices"][2], fwd["fair"], fwd["entry"]],
    })
    fig = px.bar(bars, x="Item", y="Value", text=bars["Value"].map(lambda v: f"{v:,.2f}"), title="Price vs model values")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8. RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def render_excluded(m, why):
    s = m["sym"]
    st.subheader(f"{m['name']} ({m['ticker']})")
    say("warning", f"**The screen is not applicable to the {m['sector']} sector.** {why}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", fmt_price(m["price"], s))
    c2.metric("Market cap", fmt_big(m["market_cap"], s))
    c3.metric("Trailing P/E", fmt_x(m["pe_ttm"]))
    c4.metric("Price / book", fmt_x(m["pb"], 2))
    say("info", "To run the screen anyway (results will be unreliable), tick the box in the sidebar and click Run audit again.")


def render_public(m, cfg):
    s = m["sym"]
    pillars = build_public_pillars(m)
    score, coverage, groups = summarize(pillars)
    hard, soft, flags = public_gates(m)
    kind, label, reason = decide(score, coverage, MIN_COVERAGE_PUBLIC, hard, soft)
    fwd, fwd_msg = forward_model(m, cfg)

    st.subheader(f"{m['name']} ({m['ticker']})")
    st.caption(esc(
        f"{m['sector']}, {m['industry']}. Price {fmt_price(m['price'], s)}, market cap {fmt_big(m['market_cap'], s)}. "
        f"Yahoo Finance data fetched {m['fetched_at']}."
    ))
    rule = sector_rule(m["sector"], m["industry"])
    if rule:
        extra = " You chose to run it anyway, so treat the result as unreliable." if rule[0] == "exclude" else ""
        say("warning", f"**{m['sector']} sector.** {rule[1]}{extra}")
    if m["currency_mismatch"]:
        say("warning", f"The price is in {m['currency']} but financials are reported in {m['fin_currency']}. "
                       "FCF yield and the forward valuation are skipped to avoid mixing currencies.")

    c1, c2, c3, c4 = st.columns(4)
    have = sum(1 for p in pillars if p.score is not None)
    c1.metric("Screen result", label)
    c2.metric("Composite score", "n/a" if score is None else f"{score:.2f} / 5")
    c3.metric("Data coverage", f"{coverage:.0%}",
              help=f"{have} of {len(pillars)} pillars had data. Missing pillars are left out and the remaining weights are rescaled.")
    if fwd:
        c4.metric("Model entry price", fmt_price(fwd["entry"], s),
                  delta=f"{(m['price'] / fwd['entry'] - 1) * 100:+.1f}% price vs entry", delta_color="inverse",
                  help="From the illustrative forward model in the second tab. Not an analyst target.")
    else:
        c4.metric("Model entry price", "n/a", help=fwd_msg)
    show_verdict(kind, label, reason)

    for col, (g, v) in zip(st.columns(len(groups)), groups.items()):
        col.metric(g, "n/a" if v is None else f"{v:.2f} / 5")
    show_findings(hard, soft, flags)

    t1, t2, t3 = st.tabs(["Scorecard", "Forward valuation (illustrative)", "Inputs used"])
    with t1:
        show_scorecard(pillars, f"{m['ticker']}_scorecard.csv", "Pillar scores (1 = poor, 5 = excellent)")
    with t2:
        show_forward(m, fwd, fwd_msg, cfg)
    with t3:
        st.caption("Check these against the Yahoo Finance page or the company filings before relying on the result.")
        st.dataframe(raw_inputs_table(m), hide_index=True)


def render_venture(vi, fade, cost_ratio):
    s = vi["sym"]
    d = venture_derive(vi)
    pillars = build_venture_pillars(vi, d)
    score, coverage, _ = summarize(pillars)
    hard, soft, flags = venture_gates(vi, d)
    kind, label, reason = decide(score, coverage, MIN_COVERAGE_VENTURE, hard, soft)
    can_project = (vi["revenue"] and vi["growth"] is not None and vi["gross_margin"] is not None
                   and vi["burn"] is not None)
    df_proj, summary = project_venture(vi, fade, cost_ratio) if can_project else (None, None)

    st.subheader(f"Venture growth dashboard: {vi['name']}")
    src = "Manual inputs." if vi["source"] == "manual" else f"Yahoo Finance data fetched {vi['fetched_at']}."
    st.caption(esc(f"{src} Amounts in millions of {s.strip()}."))

    rw, bm = d["runway"], d["burn_multiple"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Screen result", label)
    c2.metric("Composite score", "n/a" if score is None else f"{score:.2f} / 5")
    c3.metric("Cash runway", "n/a" if rw is None else "Self-funding" if math.isinf(rw) else f"{rw:.0f} months")
    c4.metric("Burn multiple", "n/a" if bm is None else "Self-funding" if bm == 0 else "No net new revenue"
              if math.isinf(bm) else f"{bm:.2f}x")
    c5.metric("Data coverage", f"{coverage:.0%}",
              help="Share of pillar weight that has data. Missing pillars are left out and the rest rescaled.")
    show_verdict(kind, label, reason)
    show_findings(hard, soft, flags)

    t1, t2, t3 = st.tabs(["Scorecard", "Projection and breakeven", "Inputs and derived metrics"])
    with t1:
        show_scorecard(pillars, f"{vi['name'].replace(' ', '_')}_venture_scorecard.csv", "Venture pillar scores (1 = poor, 5 = excellent)")
    with t2:
        if df_proj is None:
            say("info", "A projection needs revenue, growth, gross margin and net burn. Fill in the missing inputs.")
        else:
            st.caption("Illustrative operating model, not a forecast. Growth fades each year and the cost base grows "
                       "more slowly than revenue (operating leverage). Change both in the sidebar.")
            level, text = breakeven_message(summary)
            say(level, f"**Breakeven.** {text}")
            level, text = funding_message(summary, vi["cash"], s)
            say(level, f"**Funding.** {text}")
            long = df_proj.melt(id_vars="Year", value_vars=["Revenue", "Gross profit", "Cash costs"],
                                var_name="Series", value_name="Amount")
            fig = px.line(long, x="Year", y="Amount", color="Series", markers=True,
                          title="Revenue, gross profit and cash costs (millions)")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig)
            if not df_proj["Cash"].isna().all():
                cash_df = df_proj[["Year", "Cash"]].copy()
                cash_df["Status"] = ["Funded" if v >= 0 else "Shortfall" for v in cash_df["Cash"]]
                fig2 = px.bar(cash_df, x="Year", y="Cash", color="Status",
                              color_discrete_map={"Funded": "#0EA5E9", "Shortfall": "#DC2626"},
                              title="Projected cash balance (millions); below zero means new funding is needed")
                fig2.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig2)
            show = df_proj.copy()
            show["Year"] = ["Today"] + [f"Year {i}" for i in range(1, PROJECTION_YEARS + 1)]
            for col in ("Revenue", "Gross profit", "Cash costs", "Net burn", "Cash"):
                show[col] = show[col].map(lambda v: fmt_mm(None if math.isnan(v) else v, s))
            st.dataframe(show, hide_index=True)
    with t3:
        items = [
            ("Revenue / ARR", fmt_mm(vi["revenue"], s)),
            ("Growth (YoY)", f"{fmt_pct(vi['growth'], 0)}  [{vi.get('growth_src') or 'user input'}]"),
            ("Gross margin", fmt_pct(vi["gross_margin"])),
            ("Net revenue retention", fmt_pct(vi["nrr"], 0)),
            ("TAM", vi["tam"] or "n/a"),
            ("Cash on hand", fmt_mm(vi["cash"], s)),
            ("Debt", fmt_mm(vi.get("debt"), s)),
            ("Annual net burn (negative = cash-generative)", fmt_mm(vi["burn"], s)),
            ("Net new revenue, last 12 months (derived)", fmt_mm(d["net_new"], s)),
            ("Rule of 40: growth + free-cash-flow margin (derived)", "n/a" if d["rule40"] is None else f"{d['rule40']:.0f}"),
        ]
        st.dataframe(pd.DataFrame(items, columns=["Input", "Value"]), hide_index=True)
        if vi["source"] == "ticker":
            st.caption("Growth is the latest-quarter year-on-year rate and burn is free cash flow, both from Yahoo. "
                       "Neither is ARR or operating burn, so treat the burn multiple as an approximation.")


# ══════════════════════════════════════════════════════════════════════════════
# 9. APP FLOW
# ══════════════════════════════════════════════════════════════════════════════
def explain_error(exc):
    text = f"{type(exc).__name__}: {exc}"
    if any(k in text for k in ("RateLimit", "Too Many Requests", "429")):
        return ("Yahoo Finance is rate-limiting this server, which is common on shared cloud hosts. "
                "Wait a minute and click Run audit again. Data that was fetched successfully is cached for an hour.")
    return f"The analysis could not be completed. {text}"


def sidebar():
    sb = st.sidebar
    sb.header("Configuration")
    mode = sb.radio("Evaluation mode", [MODE_PUBLIC, MODE_VENTURE])
    cfg = {"mode": mode}
    if mode == MODE_PUBLIC:
        cfg["ticker"] = sb.text_input(
            "Ticker", value="NVDA", help="US: AAPL. India NSE: RELIANCE.NS. India BSE: 500325.BO."
        ).strip().upper()
        cfg["loss_handling"] = sb.selectbox(
            "If trailing net income is zero or negative", LOSS_OPTIONS,
            help=f"Auto sends loss-makers with revenue growth of {ROUTE_GROWTH_MIN:.0%} or more to the venture lens. "
                 "Other loss-makers stay in the screen, where a hard gate caps them at Weak.")
        cfg["allow_excluded"] = sb.checkbox(
            "Run anyway for banks, insurers and REITs", value=False,
            help="The rubric does not fit these sectors. Results will be unreliable.")
        with sb.expander("Valuation assumptions"):
            cfg["mos"] = st.slider("Margin of safety", 0.05, 0.50, 0.20, 0.05)
            cfg["req_return"] = st.slider("Required annual return", 0.08, 0.20, 0.10, 0.01)
            cfg["growth_basis"] = st.selectbox("Growth basis", list(GROWTH_BASES))
            cfg["max_growth"] = st.slider("Maximum growth assumed", 0.05, 0.40, 0.20, 0.01)
            cfg["target_peg"] = st.slider("Target PEG for the exit multiple", 0.8, 2.5, 1.5, 0.1)
            cfg["max_pe"] = st.slider("Maximum exit P/E", 12.0, 45.0, 30.0, 1.0)
    else:
        cfg["source"] = sb.radio("Data source", [SRC_MANUAL, SRC_TICKER])
        if cfg["source"] == SRC_TICKER:
            cfg["ticker"] = sb.text_input("Ticker of a loss-making or growth company", value="CRWV").strip().upper()
            cfg["nrr"] = sb.number_input("Net revenue retention % (0 = unknown)", 0, 200, 0, 5)
            cfg["tam"] = sb.selectbox("TAM (your judgement, annual USD-equivalent)", TAM_CHOICES,
                                      index=len(TAM_CHOICES) - 1)
        else:
            cfg["name"] = sb.text_input("Company name", value="My SaaS Venture").strip()
            cfg["sym"] = sb.selectbox("Currency symbol", ["$", "₹", "€", "£"])
            cfg["arr"] = sb.number_input("Current ARR / revenue (millions)", min_value=0.1, value=10.0, step=1.0)
            cfg["growth"] = sb.slider("YoY growth (%)", -20, 400, 85, 5)
            cfg["gm"] = sb.slider("Gross margin (%)", 10, 95, 75, 5)
            cfg["nrr"] = sb.slider("Net revenue retention (%)", 70, 160, 115, 5)
            cfg["tam"] = sb.selectbox("TAM (annual USD-equivalent)", TAM_CHOICES, index=0)
            cfg["cash"] = sb.number_input("Cash on hand (millions)", min_value=0.0, value=12.0, step=1.0)
            cfg["burn"] = sb.number_input("Annual net burn (millions; negative = cash-generative)", value=6.0, step=1.0)
        with sb.expander("Projection assumptions"):
            cfg["fade"] = st.slider("Growth fade per year", 0.5, 1.0, DEFAULT_FADE, 0.05,
                                    help="Growth in year t = starting growth x fade^t.")
            cfg["cost_ratio"] = st.slider("Cost growth as a share of revenue growth", 0.2, 1.0, DEFAULT_COST_RATIO, 0.05,
                                          help="Lower means more operating leverage.")
    clicked = sb.button("Run audit 🚀", type="primary")
    sb.caption("Screening tool for research and education. Not investment advice.")
    return cfg, clicked


def load(ticker):
    if not ticker:
        raise DataError("Enter a ticker symbol in the sidebar, then click Run audit.")
    with st.spinner(f"Fetching Yahoo Finance data for {ticker}..."):
        return fetch_snapshot(ticker)


def run_public(cfg):
    snap = load(cfg["ticker"])
    m = extract_public_metrics(snap)
    rule = sector_rule(m["sector"], m["industry"])
    if rule and rule[0] == "exclude" and not cfg["allow_excluded"]:
        render_excluded(m, rule[1])
        return
    if m["ni_ttm"] is not None and m["ni_ttm"] <= 0:
        growth = coalesce(m["rev_growth_yoy"], m["rev_cagr"])
        choice = cfg["loss_handling"]
        if choice == LOSS_VENTURE:
            why = "the sidebar is set to always use the venture lens"
            to_venture = True
        elif choice == LOSS_AUTO and growth is not None and growth >= ROUTE_GROWTH_MIN:
            why = f"revenue is growing {growth * 100:.0f}%"
            to_venture = True
        else:
            to_venture = False
        if to_venture:
            say("warning", f"**Routed to the venture lens.** {m['name']} has negative trailing net income and {why}, "
                           "so profit-based metrics do not apply. Switch to venture mode in the sidebar to add NRR, "
                           "TAM or change the projection assumptions.")
            render_venture(venture_inputs_from_snapshot(snap), DEFAULT_FADE, DEFAULT_COST_RATIO)
            return
        if choice == LOSS_SCREEN:
            say("warning", "**Trailing net income is negative.** The sidebar is set to always run the quality screen, "
                           "so the loss-making gate caps the result at Weak.")
        else:
            say("warning", "**Trailing net income is negative and growth is below the venture-routing threshold "
                           f"({ROUTE_GROWTH_MIN:.0%}).** The quality screen runs anyway, and the loss-making gate "
                           "caps the result at Weak.")
    render_public(m, cfg)


def run_venture(cfg):
    if cfg["source"] == SRC_TICKER:
        snap = load(cfg["ticker"])
        vi = venture_inputs_from_snapshot(
            snap,
            nrr=cfg["nrr"] / 100 if cfg["nrr"] else None,
            tam=None if cfg["tam"] == TAM_UNKNOWN else cfg["tam"],
        )
    else:
        vi = {
            "name": cfg["name"] or "Startup", "ticker": None, "sym": cfg["sym"], "source": "manual",
            "fetched_at": None, "revenue": cfg["arr"], "growth": cfg["growth"] / 100,
            "growth_src": "User input", "gross_margin": cfg["gm"] / 100, "nrr": cfg["nrr"] / 100,
            "tam": None if cfg["tam"] == TAM_UNKNOWN else cfg["tam"],
            "cash": cfg["cash"], "debt": None, "burn": cfg["burn"],
        }
    render_venture(vi, cfg["fade"], cfg["cost_ratio"])


def main():
    st.title("🛡️ Anti-Fragile Stock & Startup Screener")
    st.markdown(
        "A **15-pillar screen** for profitable listed companies and a **venture lens** for pre-profit and growth "
        "companies. Missing data is shown as *n/a* and left out of the score. It is never invented."
    )
    cfg, clicked = sidebar()
    if clicked:
        st.session_state["active_cfg"] = cfg
    active = st.session_state.get("active_cfg")

    if not active:
        say("info", "👈 Choose a mode and a target in the sidebar, then click **Run audit**.")
    else:
        if active != cfg:
            st.caption("Sidebar settings changed since the last run. Click Run audit to refresh the results.")
        try:
            if active["mode"] == MODE_PUBLIC:
                run_public(active)
            else:
                run_venture(active)
        except DataError as exc:
            say("error", str(exc))
        except Exception as exc:  # noqa: BLE001 - show any data or network problem instead of a blank page
            say("error", explain_error(exc))
            with st.expander("Technical details"):
                st.exception(exc)

    st.divider()
    with st.expander("Methodology and limits"):
        st.markdown(
            "- **Scoring:** each pillar scores 1 to 5 against fixed thresholds. The composite is the weighted average "
            "of pillars that have data; missing pillars are left out and the remaining weights are rescaled.\n"
            f"- **Coverage rule:** a result needs at least {MIN_COVERAGE_PUBLIC:.0%} of pillar weight to have data "
            f"({MIN_COVERAGE_VENTURE:.0%} in the venture lens).\n"
            f"- **Result bands:** Strong at {STRONG_MIN} or more, Mixed from {MIXED_MIN}, Weak below. "
            "A hard safety gate forces Weak and a soft gate caps at Mixed. These cut-offs are uncalibrated: "
            "run a set of known compounders, value traps and blow-ups through the screen before trusting them.\n"
            "- **Not sector-adjusted:** margin thresholds are absolute. Banks, insurers and REITs are blocked by "
            "default; utilities and cyclicals carry a caution.\n"
            "- **Forward valuation:** a transparent illustration built from conservative, capped growth, an exit "
            "multiple that can shrink but never expands, and a margin of safety. It is deliberately harsh on "
            "high-multiple growth stocks. Analyst targets are shown for reference and never used.\n"
            "- **Venture projection:** revenue growth fades each year while the cost base grows more slowly, "
            "which gives a breakeven year and a funding gap. It is a what-if tool, not a forecast.\n"
            "- **Data:** Yahoo Finance via yfinance is unofficial and can be incomplete or delayed. "
            "Compare the *Inputs used* tab with the company's filings."
        )
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
