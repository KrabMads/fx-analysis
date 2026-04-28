"""
fx_hedge_backtest_multiccy.py  —  v2
======================================
Multi-currency FX hedging backtest for a Danish company (DKK domestic).
Income currencies: GBP, SEK, USD, NOK, AUD, CAD

NEW IN v2
─────────
• Layered hedging: 1/3 notional entered at t-1M, t-2M, t-3M per cash flow.
  Smooths entry timing risk via dollar-cost averaging across entry dates.
• Carry-decomposed performance: each month's PnL split into:
    carry component   = F_entry − S_entry  (forward premium baked in at hedge entry)
    spot move         = S_settle − S_entry (what spot did over the hedge period)
    option payoff     = effective_rate − S_settle (RR collar activation, 0 for forwards)
• Crisis period analysis: COVID (Feb–May 2020), Rate Shock (2022), SVB (Mar 2023)

STRATEGIES
──────────
  Unhedged        — no hedge, receive spot at settlement
  Forward         — full notional locked at 3M forward at t-3M
  Risk Reversal   — full notional, 25Δ zero-premium collar at t-3M
  Blend 50/50     — 50% Forward + 50% RR, entered at t-3M
  Fwd Layered     — 1/3 forward at t-1M, t-2M, t-3M; blended entry rate
  RR Layered      — 1/3 RR at t-1M, t-2M, t-3M; blended payoff
  Blend Layered   — 50% Fwd + 50% RR, layered entry

DATA SOURCES (fetched live on your machine)
────────────────────────────────────────────
• FX spot      : Yahoo Finance  (e.g. GBPDKK=X)
• Volatility   : 60-day realised vol from Yahoo spot series
• Interest rates: FRED via yfinance (country 3M interbank rates)

RUN:  python fx_hedge_multiccy.py
DEMO: python fx_hedge_multiccy.py --demo
"""

import argparse
import warnings
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these values
# ─────────────────────────────────────────────────────────────────────────────

NOTIONAL_PER_CCY = {        # monthly foreign-currency notional
    "GBP": 10_000_000,
    "SEK": 100_000_000,
    "USD": 10_000_000,
    "NOK": 100_000_000,
    "AUD": 15_000_000,
    "CAD": 15_000_000,
}

BACKTEST_START  = "2020-01-01"
BACKTEST_END    = "2025-01-01"
HEDGE_TENOR_M   = 3           # months ahead hedge is entered
VOL_WINDOW      = 60          # trading days for realised vol
RR_SPREAD       = 0.005       # 25Δ RR vol spread (put vol – call vol)
N_GRID          = 300         # grid points for call strike solver
STRATEGIES      = ["Forward", "Risk Reversal", "Blend 50/50",
                   "Fwd Layered", "RR Layered", "Blend Layered"]
DOMESTIC        = "DKK"

CRISIS_PERIODS = {
    "COVID\n(Feb–May 2020)":     ("2020-02-01", "2020-05-31"),
    "Rate Shock\n(2022)":        ("2022-01-01", "2022-12-31"),
    "SVB / Banking\n(Mar 2023)": ("2023-02-01", "2023-05-31"),
}

# Yahoo Finance FX tickers  (foreign vs DKK)
YF_TICKERS = {
    "GBP": "GBPDKK=X",
    "SEK": "SEKDKK=X",
    "USD": "USDDKK=X",
    "NOK": "NOKDKK=X",
    "AUD": "AUDDKK=X",
    "CAD": "CADDKK=X",
}

# FRED series for 3-month government/policy rates (annualised %)
# These are used as the foreign risk-free rate for each currency
FRED_RATE_SERIES = {
    "USD": "DTB3",            # US 3-Month T-Bill Secondary Market Rate
    "GBP": "IR3TIB01GBM156N",# UK 3M interbank (OECD via FRED)
    "SEK": "IR3TIB01SEM156N", # Sweden 3M interbank
    "NOK": "IR3TIB01NOM156N", # Norway 3M interbank
    "AUD": "IR3TIB01AUM156N", # Australia 3M interbank
    "CAD": "IR3TIB01CAM156N", # Canada 3M interbank
    "DKK": "IRSTCI01DKM156N", # Denmark immediate rate (proxy for 3M)
}

COLORS = {
    "Forward":        "#3b82f6",
    "Risk Reversal":  "#10b981",
    "Blend 50/50":    "#f59e0b",
    "Fwd Layered":    "#93c5fd",   # lighter blue
    "RR Layered":     "#6ee7b7",   # lighter green
    "Blend Layered":  "#fcd34d",   # lighter amber
    "Unhedged":       "#64748b",
}

BG, PAN, GC, TXT, DIM = "#0a0f1e", "#111827", "#1f2937", "#e2e8f0", "#64748b"
GOOD, BAD = "#10b981", "#f87171"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fx_data(start: str, end: str) -> dict[str, pd.Series]:
    """
    Fetch daily closing FX rates from Yahoo Finance.
    Returns dict ccy -> pd.Series of closing prices (foreign/DKK).
    """
    import yfinance as yf
    fx = {}
    for ccy, ticker in YF_TICKERS.items():
        print(f"  Fetching {ticker}...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("Empty data")
            s = df["Close"].squeeze().dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            fx[ccy] = s
            print(f"OK ({len(s)} days, {s.index[0].date()}–{s.index[-1].date()})")
        except Exception as e:
            print(f"FAILED: {e}")
    return fx


def fetch_rate_data(start: str, end: str) -> dict[str, pd.Series]:
    """
    Fetch 3-month interest rates from FRED via yfinance.
    Converts from annualised % to continuous decimal.
    """
    import yfinance as yf
    rates = {}
    for ccy, series_id in FRED_RATE_SERIES.items():
        ticker = f"^{series_id}" if not series_id.startswith("^") else series_id
        # FRED tickers in yfinance format
        fred_ticker = series_id   # yfinance accepts FRED series directly
        print(f"  Fetching rate {series_id} ({ccy})...", end=" ", flush=True)
        try:
            df = yf.download(fred_ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("Empty")
            s = df["Close"].squeeze().dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            # Convert % to continuous: r_cont = ln(1 + r_pct/100)
            s = np.log(1 + s / 100)
            s = s.ffill()
            rates[ccy] = s
            print(f"OK ({s.index[0].date()}–{s.index[-1].date()})")
        except Exception as e:
            print(f"FAILED: {e} — will use fallback rate")
    return rates


def get_rate_on_date(rates: dict, ccy: str, date: pd.Timestamp,
                     fallback: float = 0.03) -> float:
    """Look up interest rate for a currency on a given date, with ffill."""
    if ccy not in rates or rates[ccy].empty:
        return fallback
    s = rates[ccy]
    # Find last available rate on or before date
    avail = s[s.index <= date]
    if avail.empty:
        return fallback
    return float(avail.iloc[-1])


def realised_vol(spot_series: pd.Series, date: pd.Timestamp,
                 window: int = VOL_WINDOW) -> float:
    """
    60-day realised vol from log returns, annualised.
    Uses all available data up to (but not including) date.
    """
    hist = spot_series[spot_series.index < date].tail(window + 1)
    if len(hist) < 10:
        return 0.08   # fallback
    log_rets = np.log(hist / hist.shift(1)).dropna()
    return float(log_rets.std() * np.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
# 2. GARMAN-KOHLHAGEN OPTION PRICER
# ─────────────────────────────────────────────────────────────────────────────

def gk_put(S, K, r_d, r_f, sigma, tau):
    sqt = np.sqrt(tau)
    d1  = (np.log(S / K) + (r_d - r_f + 0.5*sigma**2)*tau) / (sigma*sqt)
    d2  = d1 - sigma*sqt
    return K*np.exp(-r_d*tau)*norm.cdf(-d2) - S*np.exp(-r_f*tau)*norm.cdf(-d1)

def gk_call(S, K, r_d, r_f, sigma, tau):
    sqt = np.sqrt(tau)
    d1  = (np.log(S / K) + (r_d - r_f + 0.5*sigma**2)*tau) / (sigma*sqt)
    d2  = d1 - sigma*sqt
    return S*np.exp(-r_f*tau)*norm.cdf(d1) - K*np.exp(-r_d*tau)*norm.cdf(d2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZERO-PREMIUM RISK REVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def zero_premium_rr(S, r_d, r_f, sigma, tau, rr_spread=RR_SPREAD, n_grid=N_GRID):
    """
    Returns (K_put, K_call, put_premium) for a zero-premium 25Δ collar.
    """
    F         = S * np.exp((r_d - r_f) * tau)
    sigma_put = sigma + 0.5 * rr_spread
    sigma_cal = max(sigma - 0.5 * rr_spread, 1e-4)

    # 25Δ put strike (closed form)
    d1_star = -norm.ppf(0.25 * np.exp(r_f * tau))
    K_put   = S * np.exp((r_d - r_f + 0.5*sigma_put**2)*tau
                          - d1_star * sigma_put * np.sqrt(tau))

    put_prem = float(gk_put(S, K_put, r_d, r_f, sigma_put, tau))

    # Call strike via grid + linear interpolation
    xs      = np.linspace(1.0001, 2.2, n_grid)
    K_cands = xs * F
    calls   = gk_call(S, K_cands, r_d, r_f, sigma_cal, tau)
    diff    = calls - put_prem
    pos     = diff > 0
    if not pos.any():
        return K_put, F * 1.10, put_prem
    lp  = (n_grid - 1) - np.argmax(pos[::-1])
    lp  = min(lp, n_grid - 2)
    x0, x1 = xs[lp], xs[lp+1]
    d0, d1_ = diff[lp], diff[lp+1]
    denom = d1_ - d0 if abs(d1_ - d0) > 1e-15 else 1e-15
    K_call = (x0 + (x1-x0)*(-d0)/denom) * F

    return float(K_put), float(K_call), put_prem


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUSINESS DAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def first_biz_day(year: int, month: int,
                  cal: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """First business day of a given month that exists in the trading calendar."""
    first = pd.Timestamp(year, month, 1)
    last  = first + pd.offsets.MonthEnd(0)
    candidates = cal[(cal >= first) & (cal <= last)]
    return candidates[0] if len(candidates) > 0 else None


def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    m = ts.month - 1 + n
    year = ts.year + m // 12
    month = m % 12 + 1
    day = min(ts.day, pd.Timestamp(year, month, 1).days_in_month)
    return pd.Timestamp(year, month, day)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE-CURRENCY BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def _get_spot(spot: pd.Series, date: pd.Timestamp):
    avail = spot[spot.index <= date]
    return float(avail.iloc[-1]) if not avail.empty else None


def _hedge_leg(spot: pd.Series, rates: dict, ccy: str,
               entry_date: pd.Timestamp, settle_date: pd.Timestamp,
               tau: float) -> dict:
    """
    Compute all pricing inputs and payoff components for one hedge leg.
    Returns dict with: S_entry, F, K_put, K_call, sigma, r_d, r_f, carry.
    """
    S_entry = _get_spot(spot, entry_date)
    if S_entry is None:
        return None
    sigma   = max(realised_vol(spot, entry_date, VOL_WINDOW), 0.01)
    r_d     = get_rate_on_date(rates, DOMESTIC, entry_date, fallback=0.03)
    r_f     = get_rate_on_date(rates, ccy,      entry_date, fallback=0.03)
    carry   = r_d - r_f
    F       = S_entry * np.exp(carry * tau)
    try:
        K_put, K_call, _ = zero_premium_rr(S_entry, r_d, r_f, sigma, tau)
    except Exception:
        K_put, K_call = F * 0.95, F * 1.05
    return dict(S_entry=S_entry, F=F, K_put=K_put, K_call=K_call,
                sigma=sigma, r_d=r_d, r_f=r_f, carry=carry)


def run_ccy_backtest(ccy: str, spot: pd.Series, rates: dict,
                     notional: float) -> pd.DataFrame:
    """
    Full backtest for one currency pair.
    Computes all 7 strategies (unhedged + 3 single-entry + 3 layered)
    plus carry decomposition for each hedged strategy.

    LAYERED HEDGE MECHANICS
    ───────────────────────
    For a cash flow settling in month T:
      Layer 1  weight=1/3, entered at T-3M, tau=3/12
      Layer 2  weight=1/3, entered at T-2M, tau=2/12
      Layer 3  weight=1/3, entered at T-1M, tau=1/12
    Blended effective rate = weighted average of three leg rates.

    CARRY DECOMPOSITION (per strategy, per month)
    ─────────────────────────────────────────────
    For each hedge, the PnL vs unhedged is decomposed into:
      carry_comp   = F_entry − S_entry        (locked carry premium/discount)
      spot_move    = S_settle − S_entry       (spot drift over the hedge window)
      option_pnl   = effective_rate − S_settle  (RR payoff if outside strikes, else 0)

    Note: carry_comp + spot_move + option_pnl = effective_rate − S_entry
          effective_rate − S_settle = carry_comp − spot_move + option_pnl
    """
    cal    = spot.index
    tau_3m = 3 / 12.0
    records = []

    all_months = pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq="MS")

    for entry_month in all_months:
        # ── Single-entry: entered at T-3M ────────────────────────────────
        entry_date = first_biz_day(entry_month.year, entry_month.month, cal)
        if entry_date is None:
            continue
        settle_month = add_months(entry_date, HEDGE_TENOR_M)
        settle_date  = first_biz_day(settle_month.year, settle_month.month, cal)
        if settle_date is None or settle_date > cal[-1]:
            break

        S_settle = _get_spot(spot, settle_date)
        if S_settle is None:
            continue

        leg3 = _hedge_leg(spot, rates, ccy, entry_date, settle_date, tau_3m)
        if leg3 is None:
            continue

        # ── Layered: legs at T-1M, T-2M, T-3M ───────────────────────────
        layered_legs = {}
        for lag in [1, 2, 3]:
            em  = add_months(settle_date, -lag)
            ed  = first_biz_day(em.year, em.month, cal)
            if ed is None or ed > cal[-1]:
                continue
            tau_leg = lag / 12.0
            l = _hedge_leg(spot, rates, ccy, ed, settle_date, tau_leg)
            if l is not None:
                layered_legs[lag] = l

        # Blended layered effective rates (equal weight across available legs)
        w = 1 / len(layered_legs) if layered_legs else 1.0
        F_layered   = sum(w * l["F"]
                          for l in layered_legs.values())
        rr_layered  = sum(w * max(l["K_put"], min(S_settle, l["K_call"]))
                          for l in layered_legs.values())
        blend_layered = 0.5 * F_layered + 0.5 * rr_layered

        # ── Single-entry payoffs ──────────────────────────────────────────
        F           = leg3["F"]
        K_put       = leg3["K_put"]
        K_call      = leg3["K_call"]
        rr_eff      = max(K_put, min(S_settle, K_call))
        blend_eff   = 0.5 * F + 0.5 * rr_eff

        # ── DKK receipts ──────────────────────────────────────────────────
        dkk_unhedged     = notional * S_settle
        dkk_forward      = notional * F
        dkk_rr           = notional * rr_eff
        dkk_blend        = notional * blend_eff
        dkk_fwd_layered  = notional * F_layered
        dkk_rr_layered   = notional * rr_layered
        dkk_blend_layered= notional * blend_layered

        # ── Carry decomposition ───────────────────────────────────────────
        # Each component in DKK per unit of notional, then scaled
        # carry_comp : value of forward premium locked in at entry
        # spot_move  : change in spot from entry to settle
        # option_pnl : extra/less from RR collar activation
        S_entry = leg3["S_entry"]
        cc_fwd   = (F     - S_entry) * notional  # carry baked into fwd
        sm       = (S_settle - S_entry) * notional  # spot drift
        op_fwd   = 0.0                              # no option for forward
        op_rr    = (rr_eff - S_settle) * notional   # put/call activation

        # Layered: use average entry spot across legs
        S_entry_avg = sum(w * l["S_entry"] for l in layered_legs.values()) if layered_legs else S_entry
        cc_fwd_l    = (F_layered  - S_entry_avg) * notional
        op_rr_l     = (rr_layered - S_settle)    * notional

        records.append({
            # Dates
            "entry_date":         entry_date,
            "settle_date":        settle_date,
            # Market inputs
            "spot_entry":         S_entry,
            "spot_settle":        S_settle,
            "forward_rate":       F,
            "K_put":              K_put,
            "K_call":             K_call,
            "sigma":              leg3["sigma"],
            "r_d":                leg3["r_d"],
            "r_f":                leg3["r_f"],
            "carry":              leg3["carry"],
            # DKK receipts — all 7 strategies
            "dkk_unhedged":       dkk_unhedged,
            "dkk_forward":        dkk_forward,
            "dkk_rr":             dkk_rr,
            "dkk_blend":          dkk_blend,
            "dkk_fwd_layered":    dkk_fwd_layered,
            "dkk_rr_layered":     dkk_rr_layered,
            "dkk_blend_layered":  dkk_blend_layered,
            # Effective rates
            "rate_unhedged":      S_settle,
            "rate_forward":       F,
            "rate_rr":            rr_eff,
            "rate_blend":         blend_eff,
            "rate_fwd_layered":   F_layered,
            "rate_rr_layered":    rr_layered,
            "rate_blend_layered": blend_layered,
            # Carry decomposition (DKK, single-entry strategies)
            "carry_comp_fwd":     cc_fwd,          # carry locked at entry
            "spot_move_dkk":      sm,               # spot P&L (same for all)
            "option_pnl_rr":      op_rr,            # RR collar activation
            "carry_comp_fwd_l":   cc_fwd_l,         # layered carry
            "option_pnl_rr_l":    op_rr_l,          # layered RR option
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("settle_date")
    df.index = pd.to_datetime(df.index)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(dkk_series: pd.Series, strategy: str) -> dict:
    """
    Compute hedging performance metrics for a DKK cash-flow series.

    Metrics
    -------
    Mean monthly DKK     : average monthly receipt
    Total DKK            : sum over backtest period
    Std monthly DKK      : volatility of monthly receipts
    VaR-10% (monthly)    : 10th percentile monthly receipt
    CVaR-10% (monthly)   : average of worst 10% months
    Worst month          : minimum monthly receipt
    Max 3M drawdown      : worst cumulative 3-month shortfall vs S0-rate
    Mean obtained rate   : avg effective DKK/FCY rate received
    """
    s   = dkk_series.dropna()
    v10 = np.percentile(s, 10)
    # Max rolling 3-month drawdown (sum of 3 consecutive months)
    rolling3 = s.rolling(3).sum()
    max_3m_dd = float(rolling3.min()) if len(rolling3.dropna()) > 0 else np.nan

    return {
        "Strategy":           strategy,
        "Mean (M DKK)":       s.mean()   / 1e6,
        "Total (B DKK)":      s.sum()    / 1e9,
        "Std (M DKK)":        s.std()    / 1e6,
        "VaR-10% (M DKK)":   v10        / 1e6,
        "CVaR-10% (M DKK)":  s[s <= v10].mean() / 1e6,
        "Worst month (M DKK)": s.min()  / 1e6,
        "Max 3M drawdown (M)": max_3m_dd/ 1e6,
        "_series":            s,
    }


def portfolio_metrics(results: dict) -> dict[str, pd.Series]:
    """Aggregate DKK across all currencies for each strategy."""
    col_map = {
        "Unhedged":      "dkk_unhedged",
        "Forward":       "dkk_forward",
        "Risk Reversal": "dkk_rr",
        "Blend 50/50":   "dkk_blend",
        "Fwd Layered":   "dkk_fwd_layered",
        "RR Layered":    "dkk_rr_layered",
        "Blend Layered": "dkk_blend_layered",
    }
    port = {}
    for strat, col in col_map.items():
        series_list = [df[col] for df in results.values() if col in df.columns]
        if series_list:
            port[strat] = pd.concat(series_list, axis=1).sum(axis=1)
    return port


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYNTHETIC DEMO DATA  (used when --demo or no internet)
# ─────────────────────────────────────────────────────────────────────────────

# Realistic historical moments for each pair (mean, annual vol, drift per year)
DEMO_PARAMS = {
    "GBP": {"S0": 8.60,  "mu": -0.005, "sigma": 0.065, "r_f_base": 0.045},
    "SEK": {"S0": 0.65,  "mu":  0.000, "sigma": 0.080, "r_f_base": 0.020},
    "USD": {"S0": 6.90,  "mu":  0.005, "sigma": 0.070, "r_f_base": 0.045},
    "NOK": {"S0": 0.62,  "mu": -0.005, "sigma": 0.085, "r_f_base": 0.030},
    "AUD": {"S0": 4.50,  "mu": -0.010, "sigma": 0.090, "r_f_base": 0.035},
    "CAD": {"S0": 5.10,  "mu":  0.000, "sigma": 0.075, "r_f_base": 0.040},
}
DKK_RATE_BASE = 0.030   # Danish rate baseline


def generate_demo_data(start: str, end: str):
    """Generate realistic synthetic FX + rate data for demo."""
    dates = pd.bdate_range(start=start, end=end, freq="B")
    rng   = np.random.default_rng(42)
    dt    = 1 / 252

    fx_data    = {}
    rate_data  = {}

    for ccy, p in DEMO_PARAMS.items():
        # GBM spot with regime changes
        n   = len(dates)
        log_s = np.zeros(n)
        log_s[0] = np.log(p["S0"])
        # Occasional vol regimes
        regime_vol = p["sigma"]
        for i in range(1, n):
            if rng.random() < 0.003:   # 0.3% chance of vol spike per day
                regime_vol = p["sigma"] * (1.5 + rng.random())
            else:
                regime_vol = p["sigma"] + 0.3 * (p["sigma"] - regime_vol) * dt
            z = rng.standard_normal()
            log_s[i] = log_s[i-1] + (p["mu"] - 0.5 * regime_vol**2) * dt + regime_vol * np.sqrt(dt) * z
        spot = pd.Series(np.exp(log_s), index=dates, name=ccy)
        fx_data[ccy] = spot

        # OU rate process for foreign rate
        r_f = np.zeros(n)
        r_f[0] = p["r_f_base"]
        for i in range(1, n):
            r_f[i] = r_f[i-1] + 4.0 * (p["r_f_base"] - r_f[i-1]) * dt + 0.015 * np.sqrt(dt) * rng.standard_normal()
        rate_data[ccy] = pd.Series(np.maximum(r_f, 0.001), index=dates)

    # DKK rate (closely tracks ECB, with small deviations)
    n = len(dates)
    r_dkk = np.zeros(n)
    r_dkk[0] = DKK_RATE_BASE
    for i in range(1, n):
        r_dkk[i] = (r_dkk[i-1]
                    + 3.0 * (DKK_RATE_BASE - r_dkk[i-1]) * dt
                    + 0.012 * np.sqrt(dt) * rng.standard_normal())
    rate_data["DKK"] = pd.Series(np.maximum(r_dkk, 0.001), index=dates)

    return fx_data, rate_data


# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, title, fs=9):
    ax.set_facecolor(PAN)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.set_title(title, color=TXT, fontsize=fs, fontweight="bold", pad=6)
    for sp in ax.spines.values():
        sp.set_edgecolor(GC)


def plot_obtained_rates(results: dict, output_dir: str = "."):
    """Per-currency: obtained rate over time for each strategy."""
    ccys = list(results.keys())
    n    = len(ccys)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows), facecolor=BG)
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Obtained Rate Over Time  (DKK per foreign unit)",
                 color=TXT, fontsize=13, fontweight="bold", y=1.01)

    for i, ccy in enumerate(ccys):
        ax  = axes[i]
        df  = results[ccy]
        _style(ax, f"{ccy}/DKK — Effective Rate Received")

        rate_cols = {
            "Unhedged":      ("rate_unhedged", COLORS["Unhedged"]),
            "Forward":       ("rate_forward",  COLORS["Forward"]),
            "Risk Reversal": ("rate_rr",       COLORS["Risk Reversal"]),
            "Blend 50/50":   ("rate_blend",    COLORS["Blend 50/50"]),
        }

        for label, (col, col_color) in rate_cols.items():
            if col in df.columns:
                ax.plot(df.index, df[col], color=col_color,
                        lw=1.4, alpha=0.9, label=label)

        # Shade the spot rate for reference
        ax.fill_between(df.index, df["spot_settle"],
                        df["spot_settle"].mean(),
                        alpha=0.06, color="#94a3b8")
        ax.set_ylabel("DKK per FCY", color=DIM, fontsize=8)
        ax.legend(fontsize=7, labelcolor=TXT, facecolor=GC,
                  loc="upper right", framealpha=0.8)
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%Y"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_rates_over_time.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_rates_over_time.png")


def plot_dkk_cumulative(portfolio: dict[str, pd.Series], output_dir: str = "."):
    """Cumulative DKK received across all currencies, all strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), facecolor=BG)
    fig.suptitle("Portfolio DKK Cash Flows  (All Currencies Combined)",
                 color=TXT, fontsize=12, fontweight="bold")

    # ── Left: monthly DKK received ────────────────────────────────────────
    ax = axes[0]
    _style(ax, "Monthly DKK Received (all currencies)")
    for strat, s in portfolio.items():
        ax.plot(s.index, s / 1e6, color=COLORS.get(strat, "#fff"),
                lw=1.4, alpha=0.85, label=strat)
    ax.set_ylabel("DKK (millions)", color=DIM, fontsize=8)
    ax.legend(fontsize=8, labelcolor=TXT, facecolor=GC, framealpha=0.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}M"))

    # ── Right: cumulative DKK ─────────────────────────────────────────────
    ax2 = axes[1]
    _style(ax2, "Cumulative DKK Received")
    for strat, s in portfolio.items():
        ax2.plot(s.index, s.cumsum() / 1e9,
                 color=COLORS.get(strat, "#fff"),
                 lw=2, alpha=0.9, label=strat)
    ax2.set_ylabel("DKK (billions)", color=DIM, fontsize=8)
    ax2.legend(fontsize=8, labelcolor=TXT, facecolor=GC, framealpha=0.8)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}B"))

    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        ax.tick_params(axis="x", colors=TXT, labelsize=8)

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_cumulative_dkk.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_cumulative_dkk.png")


def plot_risk_dashboard(results: dict, portfolio: dict, output_dir: str = "."):
    """
    3-panel risk dashboard per strategy:
      Left:  VaR + mean bar chart (per currency)
      Middle: drawdown analysis
      Right:  summary metric table
    """
    strats_all = ["Unhedged"] + STRATEGIES
    col_map    = {
        "Unhedged":      "dkk_unhedged",
        "Forward":       "dkk_forward",
        "Risk Reversal": "dkk_rr",
        "Blend 50/50":   "dkk_blend",
        "Fwd Layered":   "dkk_fwd_layered",
        "RR Layered":    "dkk_rr_layered",
        "Blend Layered": "dkk_blend_layered",
    }

    # ── Per-currency metric tables ─────────────────────────────────────────
    ccys = list(results.keys())
    n    = len(ccys)
    fig, axes = plt.subplots(n, 3, figsize=(22, 3.8 * n + 1), facecolor=BG)
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Hedging Strategy Risk Dashboard — Per Currency",
                 color=TXT, fontsize=13, fontweight="bold", y=1.005)

    for row, ccy in enumerate(ccys):
        df = results[ccy]
        notional = NOTIONAL_PER_CCY[ccy]

        # Compute metrics for all strategies
        all_m = []
        for strat in strats_all:
            col = col_map[strat]
            if col in df.columns:
                m = compute_metrics(df[col], strat)
                all_m.append(m)

        # ── col 0: Mean + VaR-10% bar chart ──────────────────────────────
        ax0 = axes[row, 0]
        _style(ax0, f"{ccy}  —  Mean vs VaR-10% (M DKK/month)")
        xs   = np.arange(len(all_m))
        w    = 0.35
        means = [m["Mean (M DKK)"] for m in all_m]
        vars_ = [m["VaR-10% (M DKK)"] for m in all_m]
        cols_ = [COLORS.get(m["Strategy"], "#fff") for m in all_m]
        for i, (mn, vr, c) in enumerate(zip(means, vars_, cols_)):
            ax0.bar(xs[i] - w/2, mn, w, color=c, alpha=0.85, label=None)
            ax0.bar(xs[i] + w/2, vr, w, color=c, alpha=0.45, hatch="//")
        ax0.set_xticks(xs)
        ax0.set_xticklabels([m["Strategy"] for m in all_m],
                             rotation=20, ha="right", color=TXT, fontsize=7.5)
        ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}M"))
        ax0.tick_params(axis="y", colors=TXT)
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(fc="#555", label="Mean (solid)"),
                   Patch(fc="#333", hatch="//", label="VaR-10% (hatched)")]
        ax0.legend(handles=handles, fontsize=7, labelcolor=TXT,
                   facecolor=GC, framealpha=0.8)

        # ── col 1: Rolling 3-month drawdown ──────────────────────────────
        ax1 = axes[row, 1]
        _style(ax1, f"{ccy}  —  Rolling 3-Month DKK Drawdown")
        for strat in strats_all:
            col = col_map[strat]
            if col in df.columns:
                roll3 = df[col].rolling(3).sum() / 1e6
                ax1.plot(df.index, roll3,
                         color=COLORS.get(strat, "#fff"),
                         lw=1.3, alpha=0.85, label=strat)
        ax1.axhline(0, color="#94a3b8", lw=0.7, ls=":", alpha=0.5)
        ax1.set_ylabel("DKK (M)", color=DIM, fontsize=8)
        ax1.legend(fontsize=7, labelcolor=TXT, facecolor=GC,
                   loc="lower left", framealpha=0.8)
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

        # ── col 2: Summary metric table ───────────────────────────────────
        ax2 = axes[row, 2]
        _style(ax2, f"{ccy}  —  Risk Metrics")
        ax2.axis("off")

        col_hdrs = ["Mean\n(M)", "Std\n(M)", "VaR-10%\n(M)",
                    "CVaR-10%\n(M)", "Worst\nMo (M)", "Max 3M\nDD (M)"]
        keys     = ["Mean (M DKK)", "Std (M DKK)", "VaR-10% (M DKK)",
                    "CVaR-10% (M DKK)", "Worst month (M DKK)",
                    "Max 3M drawdown (M)"]
        higher   = [True, False, True, True, True, True]

        rows_t   = [[f"{m[k]:.2f}" for k in keys] for m in all_m]
        rlabs    = [m["Strategy"] for m in all_m]
        rcols_t  = [COLORS.get(s, TXT) for s in rlabs]

        # colour coding
        vals_arr = np.array([[m[k] for k in keys] for m in all_m], dtype=float)

        tbl = ax2.table(
            cellText=rows_t, rowLabels=rlabs, colLabels=col_hdrs,
            cellLoc="center", rowLoc="right", loc="center",
            bbox=[0.0, 0.02, 1.0, 0.96]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(GC)
            if r == 0:
                cell.set_facecolor("#1a2744")
                cell.set_text_props(color=TXT, fontweight="bold", fontsize=7)
            elif c == -1:
                cell.set_facecolor("#0d1220")
                cell.set_text_props(color=rcols_t[r-1], fontweight="bold")
            elif r > 0 and c >= 0:
                col_v = vals_arr[:, c]
                best  = col_v.max() if higher[c] else col_v.min()
                worst = col_v.min() if higher[c] else col_v.max()
                v     = vals_arr[r-1, c]
                if np.isclose(v, best):
                    bg, fc = "#0d2e1f", GOOD
                elif np.isclose(v, worst):
                    bg, fc = "#2e0d0d", BAD
                else:
                    bg = "#0d1220" if r % 2 == 0 else PAN
                    fc = TXT
                cell.set_facecolor(bg)
                cell.set_text_props(color=fc, fontsize=7.5)

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_risk_dashboard.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_risk_dashboard.png")


def plot_portfolio_summary(portfolio: dict, output_dir: str = "."):
    """Portfolio-level risk table + VaR distribution."""
    strats_all = ["Unhedged"] + STRATEGIES

    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    fig.suptitle("Portfolio Summary — All Currencies Combined",
                 color=TXT, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, left=0.04, right=0.98,
                           top=0.88, bottom=0.08, wspace=0.35)

    all_m = []
    for strat in strats_all:
        if strat in portfolio:
            m = compute_metrics(portfolio[strat], strat)
            all_m.append(m)

    # ── Distribution ──────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _style(ax0, "Monthly DKK Distribution (Portfolio)")
    for m in all_m:
        ax0.hist(m["_series"] / 1e6, bins=35, alpha=0.5,
                 color=COLORS.get(m["Strategy"], "#fff"),
                 density=True, histtype="stepfilled",
                 edgecolor="none", label=m["Strategy"])
        ax0.axvline(m["VaR-10% (M DKK)"],
                    color=COLORS.get(m["Strategy"], "#fff"),
                    lw=1, ls="--", alpha=0.8)
    ax0.set_xlabel("Monthly DKK (M)", color=DIM, fontsize=8)
    ax0.set_ylabel("Density", color=DIM, fontsize=8)
    ax0.legend(fontsize=8, labelcolor=TXT, facecolor=GC, framealpha=0.8)

    # ── VaR + Mean horizontal bars ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    _style(ax1, "Mean & VaR-10%  (M DKK, vs Unhedged ±)")
    uh_mean = next(m["Mean (M DKK)"] for m in all_m if m["Strategy"] == "Unhedged")
    uh_var  = next(m["VaR-10% (M DKK)"] for m in all_m if m["Strategy"] == "Unhedged")
    ys = np.arange(len(all_m))
    for i, m in enumerate(all_m):
        c = COLORS.get(m["Strategy"], "#fff")
        ax1.barh(ys[i], m["Mean (M DKK)"], color=c, alpha=0.85, height=0.35)
        delta = m["VaR-10% (M DKK)"] - uh_var
        sign  = "+" if delta >= 0 else ""
        dcol  = GOOD if delta >= 0 else BAD
        ax1.text(m["Mean (M DKK)"] + 0.3, ys[i],
                 f"VaR: {m['VaR-10% (M DKK)']:.1f}M  ({sign}{delta:.1f}M)",
                 va="center", color=dcol, fontsize=7.5)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([m["Strategy"] for m in all_m], color=TXT, fontsize=8)
    ax1.set_xlabel("Mean monthly DKK (M)", color=DIM, fontsize=8)
    ax1.axvline(uh_mean, color="#94a3b8", lw=0.8, ls="--", alpha=0.5)

    # ── Summary table ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    _style(ax2, "Portfolio Risk Table")
    ax2.axis("off")

    col_hdrs = ["Mean\n(M DKK)", "Std\n(M DKK)", "VaR-10%\n(M DKK)",
                "CVaR-10%\n(M DKK)", "Worst\nMo (M)", "Max 3M\nDD (M)"]
    keys     = ["Mean (M DKK)", "Std (M DKK)", "VaR-10% (M DKK)",
                "CVaR-10% (M DKK)", "Worst month (M DKK)",
                "Max 3M drawdown (M)"]
    higher   = [True, False, True, True, True, True]

    rows_t   = [[f"{m[k]:.2f}" for k in keys] for m in all_m]
    rlabs    = [m["Strategy"] for m in all_m]
    rcols_t  = [COLORS.get(s, TXT) for s in rlabs]
    vals_arr = np.array([[m[k] for k in keys] for m in all_m], dtype=float)

    tbl = ax2.table(
        cellText=rows_t, rowLabels=rlabs, colLabels=col_hdrs,
        cellLoc="center", rowLoc="right", loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GC)
        if r == 0:
            cell.set_facecolor("#1a2744")
            cell.set_text_props(color=TXT, fontweight="bold", fontsize=7.5)
        elif c == -1:
            cell.set_facecolor("#0d1220")
            cell.set_text_props(color=rcols_t[r-1], fontweight="bold", fontsize=8.5)
        elif r > 0 and c >= 0:
            col_v = vals_arr[:, c]
            best  = col_v.max() if higher[c] else col_v.min()
            worst = col_v.min() if higher[c] else col_v.max()
            v     = vals_arr[r-1, c]
            if np.isclose(v, best):
                bg, fc = "#0d2e1f", GOOD
            elif np.isclose(v, worst):
                bg, fc = "#2e0d0d", BAD
            else:
                bg = "#0d1220" if r % 2 == 0 else PAN
                fc = TXT
            cell.set_facecolor(bg)
            cell.set_text_props(color=fc, fontsize=8)

    plt.savefig(f"{output_dir}/fx_portfolio_summary.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_portfolio_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. CARRY DECOMPOSITION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_carry_decomposition(results: dict, output_dir: str = "."):
    """
    For each currency: stacked bar chart showing monthly DKK receipt
    decomposed into:
      Spot base     : notional × S_entry  (what you'd get at the entry-day spot)
      Carry comp    : F_entry − S_entry   (forward premium/discount locked in)
      Spot move     : S_settle − S_entry  (spot drift during the hedge window)
      Option P&L    : RR activation above/below the collar strikes
    Shown for Forward and RR Layered side-by-side.
    """
    ccys = list(results.keys())
    n    = len(ccys)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.8 * n + 0.8), facecolor=BG)
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        "Carry Decomposition  —  Monthly DKK Receipt by Component\n"
        "Spot base  +  Carry locked-in  +  Spot drift  +  Option P&L",
        color=TXT, fontsize=11, fontweight="bold", y=1.01
    )

    C_BASE  = "#334155"   # spot base (grey)
    C_CARRY = "#3b82f6"   # carry comp (blue, pos) / "#ef4444" (neg)
    C_SPOT  = "#10b981"   # spot move (green, pos) / "#f87171" (neg)
    C_OPT   = "#f59e0b"   # option P&L (amber)

    for row, ccy in enumerate(ccys):
        df = results[ccy]
        if df.empty:
            continue

        notional = NOTIONAL_PER_CCY[ccy]
        x = np.arange(len(df))
        xt = [d.strftime("%b\n%Y") for d in df.index]

        for col_idx, (title, carry_col, opt_col) in enumerate([
            ("Forward  —  carry vs spot drift",
             "carry_comp_fwd",   None),
            ("RR Layered  —  carry vs spot drift vs option",
             "carry_comp_fwd_l", "option_pnl_rr_l"),
        ]):
            ax = axes[row, col_idx]
            _style(ax, f"{ccy}  ·  {title}")

            base   = notional * df["spot_entry"] / 1e6
            carry_ = df[carry_col] / 1e6 if carry_col in df.columns else pd.Series(0, index=df.index)
            spot_m = df["spot_move_dkk"] / 1e6
            opt_   = df[opt_col] / 1e6 if (opt_col and opt_col in df.columns) else pd.Series(0, index=df.index)

            # Base
            ax.bar(x, base, color=C_BASE, alpha=0.6, label="Spot @ entry")

            # Carry comp (positive=green, negative=red)
            carry_pos = carry_.clip(lower=0)
            carry_neg = carry_.clip(upper=0)
            ax.bar(x, carry_pos, bottom=base,            color="#3b82f6", alpha=0.85, label="Carry (pos)")
            ax.bar(x, carry_neg, bottom=base,            color="#ef4444", alpha=0.85, label="Carry (neg)")

            # Spot move
            sm_pos = spot_m.clip(lower=0)
            sm_neg = spot_m.clip(upper=0)
            ax.bar(x, sm_pos, bottom=base + carry_,      color="#10b981", alpha=0.75, label="Spot move (pos)")
            ax.bar(x, sm_neg, bottom=base + carry_,      color="#f87171", alpha=0.75, label="Spot move (neg)")

            # Option P&L
            if opt_col:
                ax.bar(x, opt_.clip(lower=0),
                       bottom=base + carry_ + spot_m,    color="#f59e0b", alpha=0.9, label="Option P&L")

            # Spot settle reference line
            ax.plot(x, notional * df["spot_settle"] / 1e6,
                    color="white", lw=1.2, ls="--", alpha=0.6, label="Spot @ settle")

            ax.set_xticks(x[::3])
            ax.set_xticklabels(xt[::3], color=TXT, fontsize=7)
            ax.set_ylabel("M DKK", color=DIM, fontsize=8)
            ax.legend(fontsize=6.5, labelcolor=TXT, facecolor=GC,
                      loc="lower left", ncol=2, framealpha=0.8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}M"))

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_carry_decomposition.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_carry_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CRISIS PERIOD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_crisis_analysis(results: dict, output_dir: str = "."):
    """
    For each crisis period × currency: bar chart of mean monthly DKK
    received by strategy, and a heatmap showing strategy ranking per crisis.
    """
    col_map = {
        "Unhedged":      "dkk_unhedged",
        "Forward":       "dkk_forward",
        "Risk Reversal": "dkk_rr",
        "Blend 50/50":   "dkk_blend",
        "Fwd Layered":   "dkk_fwd_layered",
        "RR Layered":    "dkk_rr_layered",
        "Blend Layered": "dkk_blend_layered",
    }
    strats = list(col_map.keys())
    crises = list(CRISIS_PERIODS.items())
    ccys   = list(results.keys())

    # ── Fig 1: Per-currency, per-crisis bar charts ────────────────────────
    n_crisis = len(crises)
    n_ccy    = len(ccys)
    fig, axes = plt.subplots(n_crisis, n_ccy,
                             figsize=(3.8 * n_ccy, 4.2 * n_crisis + 0.8),
                             facecolor=BG)
    if n_crisis == 1: axes = axes.reshape(1, -1)
    if n_ccy    == 1: axes = axes.reshape(-1, 1)

    fig.suptitle("Crisis Period Analysis  —  Mean Monthly DKK Receipt by Strategy",
                 color=TXT, fontsize=12, fontweight="bold", y=1.01)

    for ri, (crisis_label, (cs, ce)) in enumerate(crises):
        for ci, ccy in enumerate(ccys):
            ax  = axes[ri, ci]
            df  = results[ccy]
            sub = df[(df.index >= cs) & (df.index <= ce)]

            _style(ax, f"{ccy}\n{crisis_label.replace(chr(10),' ')}", fs=7.5)

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color=DIM, transform=ax.transAxes, fontsize=9)
                continue

            notional = NOTIONAL_PER_CCY[ccy]
            means    = {s: sub[c].mean() / 1e6 for s, c in col_map.items() if c in sub.columns}
            uh_mean  = means.get("Unhedged", 0)

            ys    = np.arange(len(means))
            names = list(means.keys())
            vals  = list(means.values())
            cols_ = [COLORS.get(s, "#fff") for s in names]

            bars = ax.barh(ys, vals, color=cols_, alpha=0.82, height=0.65)

            # Delta vs unhedged annotation
            for i, (name, v) in enumerate(zip(names, vals)):
                delta = v - uh_mean
                dcol  = GOOD if delta > 0 else (BAD if delta < 0 else DIM)
                sign  = "+" if delta >= 0 else ""
                ax.text(v + abs(max(vals)-min(vals))*0.02, i,
                        f"{sign}{delta:.1f}M", va="center",
                        color=dcol, fontsize=6.5, fontweight="600")

            ax.set_yticks(ys)
            ax.set_yticklabels(names, color=TXT, fontsize=7)
            ax.axvline(uh_mean, color="#94a3b8", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel("Mean M DKK / month", color=DIM, fontsize=7)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}M"))

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_crisis_analysis.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_crisis_analysis.png")

    # ── Fig 2: Strategy ranking heatmap across crises ─────────────────────
    # Aggregate across all currencies, rank strategies per crisis
    port_crisis = {}
    for crisis_label, (cs, ce) in crises:
        strat_means = {}
        for strat, col in col_map.items():
            vals = []
            for df in results.values():
                sub = df[(df.index >= cs) & (df.index <= ce)]
                if col in sub.columns and not sub.empty:
                    vals.append(sub[col].mean())
            strat_means[strat] = np.mean(vals) / 1e6 if vals else np.nan
        port_crisis[crisis_label.replace("\n", " ")] = strat_means

    # Also add full period
    port_crisis["Full period"] = {}
    for strat, col in col_map.items():
        vals = [df[col].mean() for df in results.values() if col in df.columns]
        port_crisis["Full period"][strat] = np.mean(vals) / 1e6 if vals else np.nan

    heat_df = pd.DataFrame(port_crisis).T   # periods × strategies

    # Normalize per row (period) to show relative ranking
    heat_norm = heat_df.sub(heat_df.min(axis=1), axis=0)
    heat_norm = heat_norm.div(heat_norm.max(axis=1).replace(0, 1), axis=0)

    n_periods = len(heat_df)
    n_strats  = len(strats)
    fig2, ax  = plt.subplots(figsize=(14, 3.5), facecolor=BG)
    ax.set_facecolor(PAN)
    fig2.suptitle("Strategy Ranking Heatmap  —  Mean DKK across Periods\n"
                  "(normalized per period: green=best, red=worst)",
                  color=TXT, fontsize=11, fontweight="bold")

    for i, period in enumerate(heat_norm.index):
        for j, strat in enumerate(heat_norm.columns):
            val  = heat_norm.loc[period, strat]
            raw  = heat_df.loc[period, strat]
            # Color from red (0) to green (1)
            r = 1 - val; g = val; b = 0.3
            cell_col = f"#{int(r*180+20):02x}{int(g*180+20):02x}{int(int(b*80)):02x}"
            rect = plt.Rectangle([j-0.5, i-0.5], 1, 1,
                                  facecolor=cell_col, edgecolor=GC, lw=0.5)
            ax.add_patch(rect)
            ax.text(j, i, f"{raw:.1f}M", ha="center", va="center",
                    color="white", fontsize=8, fontweight="600")

    ax.set_xlim(-0.5, n_strats - 0.5)
    ax.set_ylim(-0.5, n_periods - 0.5)
    ax.set_xticks(range(n_strats))
    ax.set_xticklabels(list(heat_norm.columns), color=TXT, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(n_periods))
    ax.set_yticklabels(list(heat_norm.index), color=TXT, fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GC)

    plt.tight_layout(pad=1.2)
    fig2.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_crisis_heatmap.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_crisis_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11. LAYERED vs SINGLE ENTRY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_layered_comparison(results: dict, output_dir: str = "."):
    """
    Side-by-side: single-entry vs layered for each instrument type.
    Shows how layered entry smooths the obtained rate over time.
    Also shows the distribution of obtained rates (layered is tighter).
    """
    pairs = [
        ("Forward",    "Fwd Layered",  "rate_forward",    "rate_fwd_layered",  COLORS["Forward"],       COLORS["Fwd Layered"]),
        ("Risk Rev.",  "RR Layered",   "rate_rr",         "rate_rr_layered",   COLORS["Risk Reversal"], COLORS["RR Layered"]),
        ("Blend 50/50","Blend Layered","rate_blend",       "rate_blend_layered",COLORS["Blend 50/50"],   COLORS["Blend Layered"]),
    ]
    ccys = list(results.keys())
    n    = len(ccys)

    fig, axes = plt.subplots(n, len(pairs),
                             figsize=(6.5 * len(pairs), 3.6 * n + 0.8),
                             facecolor=BG)
    if n == 1: axes = axes.reshape(1, -1)
    fig.suptitle(
        "Single Entry (3M) vs Layered Entry (1/3 each at 1M, 2M, 3M)\n"
        "Obtained DKK rate per unit of foreign currency",
        color=TXT, fontsize=11, fontweight="bold", y=1.01
    )

    for row, ccy in enumerate(ccys):
        df = results[ccy]
        if df.empty:
            continue
        for col_idx, (s1, s2, rc1, rc2, c1, c2) in enumerate(pairs):
            ax = axes[row, col_idx]
            _style(ax, f"{ccy}  ·  {s1} vs {s2}")

            if rc1 in df.columns:
                ax.plot(df.index, df[rc1], color=c1, lw=1.5, alpha=0.9,
                        label=f"{s1}")
            if rc2 in df.columns:
                ax.plot(df.index, df[rc2], color=c2, lw=1.5, alpha=0.9,
                        ls="--", label=f"{s2}")

            # Spot reference
            ax.plot(df.index, df["spot_settle"], color="#94a3b8",
                    lw=0.8, alpha=0.4, ls=":", label="Spot")

            ax.set_ylabel("DKK per FCY", color=DIM, fontsize=8)
            ax.legend(fontsize=7.5, labelcolor=TXT, facecolor=GC, framealpha=0.8)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

            # Annotation: std comparison
            if rc1 in df.columns and rc2 in df.columns:
                std1 = df[rc1].std()
                std2 = df[rc2].std()
                improvement = (std1 - std2) / std1 * 100
                col = GOOD if improvement > 0 else BAD
                ax.text(0.02, 0.04,
                        f"Std: {s1} {std1:.4f}  →  {s2} {std2:.4f}  "
                        f"({'−' if improvement>0 else '+'}{abs(improvement):.1f}% vol)",
                        transform=ax.transAxes, color=col,
                        fontsize=7, fontfamily="monospace")

    plt.tight_layout(pad=1.5)
    fig.patch.set_facecolor(BG)
    plt.savefig(f"{output_dir}/fx_layered_comparison.png",
                dpi=145, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved: fx_layered_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12. EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_results(results: dict, portfolio: dict, output_dir: str = "."):
    """Save per-currency and portfolio CSVs."""
    for ccy, df in results.items():
        out = df.drop(columns=[c for c in df.columns if c.startswith("_")],
                      errors="ignore")
        out.to_csv(f"{output_dir}/hedge_{ccy}_DKK.csv", float_format="%.5f")
    print(f"  Saved {len(results)} per-currency CSVs")

    # Portfolio
    port_df = pd.DataFrame(portfolio)
    port_df.to_csv(f"{output_dir}/hedge_portfolio.csv", float_format="%.0f")
    print("  Saved: hedge_portfolio.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-currency FX hedge backtest")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no internet required)")
    parser.add_argument("--start",   default=BACKTEST_START)
    parser.add_argument("--end",     default=BACKTEST_END)
    parser.add_argument("--outdir",  default="/mnt/user-data/outputs")
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print("  Multi-Currency FX Hedge Backtest  —  Danish Company (DKK)")
    print(f"{'═'*65}")
    print(f"  Currencies : {', '.join(NOTIONAL_PER_CCY.keys())}")
    print(f"  Period     : {args.start}  →  {args.end}")
    print(f"  Tenor      : {HEDGE_TENOR_M} months")
    print(f"  Vol window : {VOL_WINDOW} trading days")
    print(f"  Strategies : {', '.join(STRATEGIES)}")
    print(f"  Mode       : {'DEMO (synthetic data)' if args.demo else 'LIVE (Yahoo Finance)'}")

    # ── Fetch data ────────────────────────────────────────────────────────
    if args.demo:
        print("\n[1/4] Generating synthetic demo data...")
        fx_data, rate_data = generate_demo_data(args.start, args.end)
    else:
        print("\n[1/4] Fetching FX data from Yahoo Finance...")
        fx_data = fetch_fx_data(args.start, args.end)
        if not fx_data:
            print("  No FX data fetched. Run with --demo or check internet access.")
            sys.exit(1)
        print("\n  Fetching interest rate data from FRED...")
        rate_data = fetch_rate_data(args.start, args.end)

    # ── Run backtest ──────────────────────────────────────────────────────
    print("\n[2/4] Running backtests...")
    results = {}
    for ccy in NOTIONAL_PER_CCY:
        if ccy not in fx_data:
            print(f"  {ccy}: no spot data — skipping")
            continue
        print(f"  {ccy}...", end=" ", flush=True)
        df = run_ccy_backtest(
            ccy      = ccy,
            spot     = fx_data[ccy],
            rates    = rate_data,
            notional = NOTIONAL_PER_CCY[ccy],
        )
        if df.empty:
            print("no trades generated")
            continue
        results[ccy] = df
        print(f"  {len(df)} monthly hedges  "
              f"| avg spot {df['spot_entry'].mean():.4f}"
              f"  vol {df['sigma'].mean()*100:.1f}%"
              f"  carry {df['carry'].mean()*100:+.2f}%")

    if not results:
        print("No results — exiting.")
        sys.exit(1)

    # ── Portfolio aggregation ─────────────────────────────────────────────
    portfolio = portfolio_metrics(results)

    # ── Console summary ───────────────────────────────────────────────────
    print("\n[3/4] Portfolio summary:")
    for strat in ["Unhedged"] + STRATEGIES:
        if strat in portfolio:
            m = compute_metrics(portfolio[strat], strat)
            print(f"  {strat:<16} "
                  f"Mean={m['Mean (M DKK)']:.1f}M  "
                  f"Std={m['Std (M DKK)']:.1f}M  "
                  f"VaR10%={m['VaR-10% (M DKK)']:.1f}M  "
                  f"MaxDD3M={m['Max 3M drawdown (M)']:.1f}M")

    # ── Plots + export ────────────────────────────────────────────────────
    print(f"\n[4/4] Generating charts → {args.outdir}/")
    plot_obtained_rates(results, args.outdir)
    plot_dkk_cumulative(portfolio, args.outdir)
    plot_risk_dashboard(results, portfolio, args.outdir)
    plot_portfolio_summary(portfolio, args.outdir)
    plot_carry_decomposition(results, args.outdir)
    plot_crisis_analysis(results, args.outdir)
    plot_layered_comparison(results, args.outdir)
    export_results(results, portfolio, args.outdir)

    print(f"\n  All done. Outputs in {args.outdir}/\n")


if __name__ == "__main__":
    main()
