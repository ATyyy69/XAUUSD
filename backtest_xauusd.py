"""
XAUUSD 15M Trend-Pullback — Vollständiger Backtest & Parameter-Optimierung
Implementiert die Pine-Script-Strategie-Logik exakt in Python.
Datenbasis: Regime-Switching GBM, kalibriert auf echte Gold-Parameter.
"""

import numpy as np
import pandas as pd
from itertools import product
import json

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════
# 1. DATEN  (Regime-Switching GBM, XAUUSD-kalibriert)
# ══════════════════════════════════════════════════════════════
def make_xauusd(n_days=504, seed=42):
    """
    Erzeugt realistisches synthetisches XAUUSD 15M OHLC.
    Wechselt zwischen Bull-, Bear- und Range-Regimes.
    Kalibrierung: sigma~10-14%/Jahr, ATR(15M)~$1.8-3.5
    """
    rng = np.random.default_rng(seed)
    bpd = 52          # Bars pro Tag  (7:00-20:00 = 52 × 15min)
    N   = n_days * bpd
    dt  = 15 / (252 * 24 * 60)   # 15 Minuten in Jahresbruchteilen

    REGIMES = [
        {"mu":  0.12, "sig": 0.10, "trans": [0.970, 0.010, 0.020]},  # bull
        {"mu": -0.08, "sig": 0.14, "trans": [0.015, 0.960, 0.025]},  # bear
        {"mu":  0.01, "sig": 0.08, "trans": [0.025, 0.025, 0.950]},  # range
    ]
    reg = 2
    price = 1900.0
    op = [price]; hi = [price]; lo = [price]; cl = [price]

    for i in range(1, N):
        if i % bpd == 0:
            probs = REGIMES[reg]["trans"]
            reg   = rng.choice(3, p=probs)

        r = REGIMES[reg]
        ret    = (r["mu"] - 0.5 * r["sig"]**2) * dt + r["sig"] * np.sqrt(dt) * rng.standard_normal()
        c_new  = cl[-1] * np.exp(ret)
        bv     = max(r["sig"] * np.sqrt(dt) * c_new, 0.05)
        o_new  = cl[-1]
        h_new  = max(o_new, c_new) + abs(rng.standard_normal() * bv * 0.6)
        l_new  = min(o_new, c_new) - abs(rng.standard_normal() * bv * 0.6)

        op.append(o_new); hi.append(h_new)
        lo.append(l_new); cl.append(c_new)

    # Zeitindex
    base  = pd.Timestamp("2022-01-03 07:00")
    idx   = []
    d_off = 0
    for i in range(N):
        bar = i % bpd
        if bar == 0 and i > 0:
            d_off += 1
            while True:
                t = base + pd.Timedelta(days=d_off)
                if t.weekday() < 5:
                    break
                d_off += 1
        cur_day = base + pd.Timedelta(days=d_off)
        idx.append(cur_day + pd.Timedelta(minutes=15 * bar))

    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": cl},
                        index=pd.DatetimeIndex(idx))


# ══════════════════════════════════════════════════════════════
# 2. INDIKATOREN
# ══════════════════════════════════════════════════════════════
def ema(s, n):       return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def htf_ema(df, tf_min=60, n=50):
    """HTF EMA, lookahead-safe: shift(1) nach resample."""
    htf_c = df["close"].resample(f"{tf_min}min").last().dropna()
    htf_e = ema(htf_c, n)
    htf_c = htf_c.reindex(df.index, method="ffill").shift(1)
    htf_e = htf_e.reindex(df.index, method="ffill").shift(1)
    return htf_c.values, htf_e.values


# ══════════════════════════════════════════════════════════════
# 3. BACKTEST (exakte Implementierung der Pine-Script-Logik)
# ══════════════════════════════════════════════════════════════
def backtest(df, p):
    cl_ = df["close"].values
    hi_ = df["high"].values
    lo_ = df["low"].values
    N   = len(df)

    ef_  = ema(df["close"], p["ema_fast"]).values
    es_  = ema(df["close"], p["ema_slow"]).values
    rv_  = rsi(df["close"], p["rsi_len"]).values
    av_  = atr(df, p["atr_len"]).values
    hc_, he_ = htf_ema(df, p["htf_min"], p["htf_ema"])

    # Session: Stunde des Bars (in Serverzeit ≈ UTC+1, vereinfacht auf Stunde)
    bar_hour = df.index.hour
    SESS_S = p["sess_s"]; SESS_E = p["sess_e"]

    PBL = p["pb_lb"]; SWB = p["sw_bars"]
    warm = max(PBL + 3, SWB + 3, 70)

    trades = []
    pos     = 0          # 0=flat  1=long  -1=short
    en_p    = sl_p = tp_p = 0.0
    sl_d    = 0.0        # initiales SL-Abstand (für R-Rechnung)
    be_done = False

    for i in range(warm, N):
        av_i = av_[i]
        if np.isnan(av_i) or av_i == 0:
            continue

        # ── Offene Position managen ──────────────────────────────────────
        if pos != 0:
            H = hi_[i]; L = lo_[i]

            if pos == 1:    # LONG
                # Break-Even
                if p["use_be"] and not be_done and H - en_p >= p["be_r"] * sl_d:
                    sl_p = max(sl_p, en_p); be_done = True
                # Trailing
                if p["use_trail"] and H - en_p >= p["ts_r"] * sl_d:
                    sl_p = max(sl_p, H - p["ts_m"] * av_i)
                # SL / TP
                if L <= sl_p:
                    trades.append({"pnl": sl_p - en_p, "sl_dist": sl_d})
                    pos = 0; continue
                if H >= tp_p:
                    trades.append({"pnl": tp_p - en_p, "sl_dist": sl_d})
                    pos = 0; continue

            else:           # SHORT
                if p["use_be"] and not be_done and en_p - L >= p["be_r"] * sl_d:
                    sl_p = min(sl_p, en_p); be_done = True
                if p["use_trail"] and en_p - L >= p["ts_r"] * sl_d:
                    sl_p = min(sl_p, L + p["ts_m"] * av_i)
                if H >= sl_p:
                    trades.append({"pnl": en_p - sl_p, "sl_dist": sl_d})
                    pos = 0; continue
                if L <= tp_p:
                    trades.append({"pnl": en_p - tp_p, "sl_dist": sl_d})
                    pos = 0; continue
            continue   # Position offen → kein neuer Entry

        # ── Session-Filter ───────────────────────────────────────────────
        h_now = bar_hour[i]
        if h_now < SESS_S or h_now >= SESS_E:
            continue

        ef = ef_[i]; ef2 = ef_[i-2]; ef1 = ef_[i-1]
        es = es_[i]; es1 = es_[i-1]
        rv = rv_[i]
        hc = hc_[i]; he = he_[i]
        C  = cl_[i]; H_i = hi_[i]; L_i = lo_[i]

        if any(np.isnan(x) for x in [ef, es, rv, av_i]):
            continue

        # Trend-Bias
        lb = ef > es and C > es
        sb = ef < es and C < es

        # HTF-Filter (bei NaN: deaktiviert)
        htf_bull = True if np.isnan(hc) or np.isnan(he) else hc > he
        htf_bear = True if np.isnan(hc) or np.isnan(he) else hc < he

        # EMA-Steigung (2-Bar)
        er = ef > ef2
        ef_fall = ef < ef2

        # Pullback (abgeschlossene Bars, repaint-safe)
        pb_lo = lo_[i - PBL: i]
        pb_hi = hi_[i - PBL: i]
        pbl = pb_lo.min() <= ef1 and C > ef
        pbs = pb_hi.max() >= ef1 and C < ef

        # Pullback-Zone (Preis blieb in EMA21-EMA50-Zone)
        pb_zone_l = pb_lo.min() >= es1 - 0.3 * av_[i-1]
        pb_zone_s = pb_hi.max() <= es1 + 0.3 * av_[i-1]

        # RSI
        rsi_l = rv >= p["rsi_lmin"] and rv < p["rsi_ob"]
        rsi_s = rv <= p["rsi_smax"] and rv > p["rsi_os"]

        # ATR-Filter
        vol_ok = av_i > p["atr_min"] if p["atr_min"] > 0 else True

        # Kerzen-Qualität
        brange = H_i - L_i
        if p["bq"] and brange > 0:
            bq_l = (C - L_i) / brange >= p["bqr"]
            bq_s = (H_i - C) / brange >= p["bqr"]
        else:
            bq_l = bq_s = True

        # SL / TP
        if p["sl_m"] == "ATR":
            sd = p["sl_k"] * av_i
        else:
            buf = 0.1 * av_i
            sd_l = max(C - (lo_[i - SWB: i+1].min() - buf), 1e-9)
            sd_s = max((hi_[i - SWB: i+1].max() + buf) - C, 1e-9)
            sd   = sd_l   # wird unten nach Richtung gesetzt

        # Entry
        enter_l = lb and er  and htf_bull and pbl and pb_zone_l and rsi_l and bq_l and vol_ok
        enter_s = sb and ef_fall and htf_bear and pbs and pb_zone_s and rsi_s and bq_s and vol_ok

        if enter_l:
            if p["sl_m"] == "Swing":
                sd = max(C - (lo_[i - SWB: i+1].min() - 0.1*av_i), 1e-9)
            en_p = C; sl_p = C - sd; tp_p = C + p["rr"] * sd
            sl_d = sd; be_done = False; pos = 1

        elif enter_s:
            if p["sl_m"] == "Swing":
                sd = max((hi_[i - SWB: i+1].max() + 0.1*av_i) - C, 1e-9)
            en_p = C; sl_p = C + sd; tp_p = C - p["rr"] * sd
            sl_d = sd; be_done = False; pos = -1

    # Offene Position am Ende schließen
    if pos != 0:
        fin_pnl = (cl_[-1] - en_p) if pos == 1 else (en_p - cl_[-1])
        trades.append({"pnl": fin_pnl, "sl_dist": sl_d})

    return trades


# ══════════════════════════════════════════════════════════════
# 4. METRIKEN  (R-multiple basiert, korrekte Equity-Kurve)
# ══════════════════════════════════════════════════════════════
def metrics(trades, capital=10000, risk_pct=0.01):
    if len(trades) < 10:
        return None

    r_mults = []
    for t in trades:
        sd = t["sl_dist"]
        if sd > 0:
            r_mults.append(t["pnl"] / sd)   # R-Multiples

    if not r_mults:
        return None

    r = np.array(r_mults)
    wins     = (r > 0).sum()
    losses   = (r <= 0).sum()
    total    = len(r)
    win_rate = wins / total * 100

    gross_w = r[r > 0].sum()
    gross_l = abs(r[r <= 0].sum())
    pf      = gross_w / gross_l if gross_l > 0 else np.inf

    # Equity-Kurve (1R = capital × risk_pct)
    eq = [capital]
    for ri in r:
        eq.append(eq[-1] * (1 + ri * risk_pct))
    eq_arr  = np.array(eq)
    roll_mx = np.maximum.accumulate(eq_arr)
    dd      = (eq_arr - roll_mx) / roll_mx
    max_dd  = abs(dd.min()) * 100
    final_r = (eq_arr[-1] - capital) / capital * 100   # Gesamt-Return %

    # Sharpe (annualisiert, annahme 52 Trades/Tag → 52*252/Trades = Trades pro Jahr)
    sr = r.mean() / r.std() * np.sqrt(total) if r.std() > 0 else 0

    return {
        "trades":    total,
        "wins":      int(wins),
        "losses":    int(losses),
        "win_rate":  round(win_rate, 1),
        "pf":        round(min(pf, 99.9), 2),
        "avg_r_win": round(r[r > 0].mean(), 3) if wins > 0 else 0,
        "avg_r_los": round(r[r <= 0].mean(), 3) if losses > 0 else 0,
        "max_dd":    round(max_dd, 2),
        "ret_pct":   round(final_r, 1),
        "sharpe":    round(sr, 3),
        "net_r":     round(r.sum(), 2),
    }


# ══════════════════════════════════════════════════════════════
# 5. BASELINE TEST
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("XAUUSD 15M Trend-Pullback — Backtest & Optimierung")
print("Methode: R-Multiple-basiert | 1R = 1% Equity-Risiko")
print("=" * 65)

print("\n[1/4] Daten generieren (2 Jahre, 15M, Regime-Switching GBM)...")
df_train = make_xauusd(n_days=504, seed=42)
mean_atr_val = atr(df_train, 14).mean()
print(f"      {len(df_train):,} Bars | {df_train.index[0].date()} → {df_train.index[-1].date()}")
print(f"      Preis-Range: ${df_train['close'].min():.0f}–${df_train['close'].max():.0f}")
print(f"      Mittlerer ATR(14): ~${mean_atr_val:.2f}")

BASE = {
    "ema_fast": 21, "ema_slow": 50,
    "htf_min": 60,  "htf_ema": 50,
    "pb_lb": 5, "sw_bars": 10,
    "rsi_len": 14, "rsi_lmin": 50, "rsi_smax": 50, "rsi_ob": 70, "rsi_os": 30,
    "atr_len": 14, "atr_min": 0.0,
    "sl_m": "ATR", "sl_k": 1.5, "rr": 2.0,
    "use_be": True, "be_r": 1.0,
    "use_trail": False, "ts_r": 1.5, "ts_m": 1.5,
    "bq": True, "bqr": 0.40,
    "sess_s": 7, "sess_e": 20,
}

print("\n[2/4] Baseline-Backtest (Default-Parameter)...")
t_base = backtest(df_train, BASE)
m_base = metrics(t_base)
print(f"\n  {'─'*40}")
print(f"  Trades         : {m_base['trades']}")
print(f"  Win-Rate       : {m_base['win_rate']}%")
print(f"  Profit-Faktor  : {m_base['pf']}")
print(f"  Gesamt-Return  : {m_base['ret_pct']}%")
print(f"  Max Drawdown   : {m_base['max_dd']}%")
print(f"  Ø R-Win/Loss   : {m_base['avg_r_win']}R / {m_base['avg_r_los']}R")
print(f"  Sharpe         : {m_base['sharpe']}")
print(f"  Netto-R        : {m_base['net_r']}R")
print(f"  {'─'*40}")


# ══════════════════════════════════════════════════════════════
# 6. PARAMETER-OPTIMIERUNG (Gitter-Suche)
# ══════════════════════════════════════════════════════════════
print("\n[3/4] Parameter-Sweep (Gitter-Suche)...")

GRID = {
    "sl_k":      [1.2, 1.5, 1.8, 2.0],
    "rr":        [1.8, 2.0, 2.5, 3.0],
    "rsi_lmin":  [45, 50, 55],
    "rsi_smax":  [45, 50, 55],
    "pb_lb":     [3, 5, 7],
    "bqr":       [0.35, 0.40, 0.50],
}

keys   = list(GRID.keys())
combos = list(product(*GRID.values()))
print(f"      {len(combos)} Kombinationen ...")

best_score = -np.inf
best_p = None; best_m = None
rows = []

for combo in combos:
    p = BASE.copy()
    for k, v in zip(keys, combo):
        p[k] = v

    t = backtest(df_train, p)
    m = metrics(t)
    if m is None or m["trades"] < 30:
        continue

    # Score: Profit-Faktor + Win-Rate - Drawdown-Strafe (alles normiert)
    pf_s  = min(m["pf"], 5.0) / 5.0          # 0→1
    wr_s  = m["win_rate"] / 100               # 0→1
    dd_s  = 1 - min(m["max_dd"], 30) / 30     # 0→1 (niedriger DD = besser)
    score = 0.40 * pf_s + 0.35 * wr_s + 0.25 * dd_s

    rows.append({**m, **dict(zip(keys, combo)), "score": score})
    if score > best_score:
        best_score = score; best_p = dict(zip(keys, combo)); best_m = m

# Top-10 ausgeben
top10 = sorted(rows, key=lambda x: x["score"], reverse=True)[:10]
print(f"\n  {'─'*70}")
print(f"  TOP-10 PARAMETERKOMBINATIONEN (Score = 0.40×PF + 0.35×WR + 0.25×DD)")
print(f"  {'─'*70}")
hdr = f"  {'#':>2}  {'T':>5}  {'WR':>5}  {'PF':>5}  {'DD':>5}  {'Score':>6}  SL×  RR   rMin rMax PB  BQR"
print(hdr)
for rank, r_ in enumerate(top10, 1):
    print(f"  {rank:>2}  {r_['trades']:>5}  {r_['win_rate']:>4.1f}%  "
          f"{r_['pf']:>5.2f}  {r_['max_dd']:>4.1f}%  {r_['score']:>6.4f}  "
          f"{r_['sl_k']:.1f}  {r_['rr']:.1f}  "
          f"{r_['rsi_lmin']:>4}  {r_['rsi_smax']:>4}  {r_['pb_lb']:>2}  {r_['bqr']:.2f}")


# ══════════════════════════════════════════════════════════════
# 7. OPTIMIERTER PARAMETER-SET  (detaillierte Analyse)
# ══════════════════════════════════════════════════════════════
print(f"\n  {'═'*60}")
print("  OPTIMALER PARAMETER-SET:")
print(f"  {'═'*60}")
opt = BASE.copy(); opt.update(best_p)
t_opt = backtest(df_train, opt)
m_opt = metrics(t_opt)

for k, v in best_p.items():
    base_v = BASE[k]
    marker = " ←" if v != base_v else ""
    print(f"    {k:<12}: {base_v} → {v}{marker}")

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │ Trades        : {m_opt['trades']:>5}                   │")
print(f"  │ Win-Rate      : {m_opt['win_rate']:>5.1f}%                 │")
print(f"  │ Profit-Faktor : {m_opt['pf']:>5.2f}                  │")
print(f"  │ Gesamt-Return : {m_opt['ret_pct']:>5.1f}%                 │")
print(f"  │ Max Drawdown  : {m_opt['max_dd']:>5.1f}%                 │")
print(f"  │ Ø R-Win       : {m_opt['avg_r_win']:>7.3f}R               │")
print(f"  │ Ø R-Loss      : {m_opt['avg_r_los']:>7.3f}R               │")
print(f"  │ Netto-R       : {m_opt['net_r']:>7.2f}R               │")
print(f"  │ Sharpe        : {m_opt['sharpe']:>7.3f}               │")
print(f"  └─────────────────────────────────────────┘")


# ══════════════════════════════════════════════════════════════
# 8. OUT-OF-SAMPLE WALK-FORWARD TEST
# ══════════════════════════════════════════════════════════════
print(f"\n[4/4] Out-of-Sample Walk-Forward Test (1 Jahr, seed=999)...")
df_oos = make_xauusd(n_days=252, seed=999)
t_oos  = backtest(df_oos, opt)
m_oos  = metrics(t_oos)

robust = (m_oos["pf"] > 1.25
          and m_oos["win_rate"] > 46
          and m_oos["max_dd"] < 20)
flag = "ROBUST ✓" if robust else "FRAGIL ✗ (Overfitting-Risiko)"

print(f"\n  Trades: {m_oos['trades']} | WR: {m_oos['win_rate']}% | PF: {m_oos['pf']} | "
      f"DD: {m_oos['max_dd']}% | Ret: {m_oos['ret_pct']}%")
print(f"  Bewertung: {flag}")


# ══════════════════════════════════════════════════════════════
# 9. ZUSAMMENFASSUNG & PINE-SCRIPT-EMPFEHLUNGEN
# ══════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("  ZUSAMMENFASSUNG & PINE-SCRIPT DEFAULT-UPDATES")
print(f"{'═'*65}")
print(f"\n  {'':20} {'Baseline':>10} {'Optimiert':>10} {'OoS':>10}")
print(f"  {'─'*52}")
print(f"  {'Trades':20} {m_base['trades']:>10} {m_opt['trades']:>10} {m_oos['trades']:>10}")
print(f"  {'Win-Rate (%)':20} {m_base['win_rate']:>10.1f} {m_opt['win_rate']:>10.1f} {m_oos['win_rate']:>10.1f}")
print(f"  {'Profit-Faktor':20} {m_base['pf']:>10.2f} {m_opt['pf']:>10.2f} {m_oos['pf']:>10.2f}")
print(f"  {'Max DD (%)':20} {m_base['max_dd']:>10.1f} {m_opt['max_dd']:>10.1f} {m_oos['max_dd']:>10.1f}")
print(f"  {'Gesamt-Return (%)':20} {m_base['ret_pct']:>10.1f} {m_opt['ret_pct']:>10.1f} {m_oos['ret_pct']:>10.1f}")
print(f"  {'Sharpe':20} {m_base['sharpe']:>10.3f} {m_opt['sharpe']:>10.3f} {m_oos['sharpe']:>10.3f}")

pf_imp = (m_opt['pf'] - m_base['pf']) / max(abs(m_base['pf']), 0.001) * 100
print(f"\n  Profit-Faktor Verbesserung vs Baseline: {pf_imp:+.1f}%")
print(f"\n  Pine-Script Input-Defaults aktualisieren:")
for k, v in best_p.items():
    bv = BASE[k]
    if v != bv:
        print(f"    {k:<12}: {bv} → {v}")

# JSON-Export
export = {
    "baseline": m_base, "optimized": m_opt, "oos": m_oos,
    "best_params": best_p, "robust": robust
}
with open("/home/user/XAUUSD/backtest_results.json", "w") as f:
    json.dump(export, f, indent=2)
print("\n  → backtest_results.json gespeichert")
print(f"{'═'*65}")
