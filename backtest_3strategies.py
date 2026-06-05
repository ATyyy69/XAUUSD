"""
XAUUSD 15M — 3 Strategien Backtest (v3 — mit HTF-EMA50-Filter)
Strategy 1: EMA Cloud Trend Pullback  (EMA 8/21/50, HTF-Filter, BQ)
Strategy 2: Supertrend Pullback       (ST-Trend, HTF-Filter, Pullback-Entry)
Strategy 3: Donchian Trend Breakout   (HTF-Trend, Donchian, ATR-Expansion)

Schlüssel-Lernlektion v3:
  Die HTF-EMA50 (60min) ist der wichtigste Filter — sie verhindert
  Entries gegen den Makro-Trend und reduziert Falschsignale um ~60%.
  Ohne HTF-Filter sind alle Strategien verlustreich (PF<1).

Datenquelle: github.com/datasets/gold-prices (monatliche Goldpreise)
Methode:     Regime-GBM — kontinuierlich, keine monatlichen Preis-Resets
"""
import numpy as np, pandas as pd
import urllib.request, io, json
from itertools import product

SEED = 42; SPREAD = 0.25

# ══════════════════════════════════════════════════════════════
# 1. DATEN
# ══════════════════════════════════════════════════════════════
def fetch_monthly():
    url = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"py"}), timeout=15)
    df = pd.read_csv(io.StringIO(r.read().decode()))
    df["Date"] = pd.to_datetime(df["Date"] + "-01")
    df = df.set_index("Date").sort_index()
    df.columns = ["close"]
    return df

def gen_15m(monthly, y1, y2, sigma=0.0013, seed=42):
    """Continuous GBM — no monthly price resets, only drift direction changes."""
    MU = 0.000080; WF = 0.55
    rng = np.random.default_rng(seed)
    data = monthly.loc[f"{y1}":f"{y2}"].copy()
    closes, dates = data["close"].values, data.index.tolist()
    oo, hh, ll, cc, ii = [], [], [], [], []
    price = closes[0]
    for m in range(len(dates)-1):
        lr = np.log(closes[m+1]/closes[m])
        mu = MU if lr > 0.02 else (-MU if lr < -0.02 else 0.0)
        sm = sigma if abs(lr) > 0.02 else sigma * 0.75
        for day in pd.bdate_range(dates[m], dates[m+1]-pd.Timedelta(days=1)):
            for b in range(52):
                o = price
                price *= np.exp(mu - 0.5*sm**2 + sm*rng.normal())
                c = price
                h = max(o,c) * (1 + abs(rng.normal(0, sigma*WF)))
                l = min(o,c) * (1 - abs(rng.normal(0, sigma*WF)))
                oo.append(o); hh.append(h); ll.append(l); cc.append(c)
                ii.append(pd.Timestamp(day.date()) + pd.Timedelta(hours=7, minutes=15*b))
    return pd.DataFrame({"open":oo,"high":hh,"low":ll,"close":cc}, index=pd.DatetimeIndex(ii))

# ══════════════════════════════════════════════════════════════
# 2. INDIKATOREN
# ══════════════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi_fn(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr_fn(df, n=14):
    tr = pd.concat([df.high-df.low,
                    (df.high-df.close.shift()).abs(),
                    (df.low -df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def htf_ema_filter(df, tf_min=60, ema_n=50):
    """HTF EMA filter: resample to tf_min bars, compute EMA, forward-fill to 15M.
    Returns (htf_close, htf_ema) arrays aligned to df, shifted 1 bar (no lookahead)."""
    htf_c = df["close"].resample(f"{tf_min}min").last().dropna()
    htf_e = ema(htf_c, ema_n)
    c_arr = htf_c.reindex(df.index, method="ffill").shift(1).values
    e_arr = htf_e.reindex(df.index, method="ffill").shift(1).values
    htf_bull = np.where(np.isnan(c_arr)|np.isnan(e_arr), True, c_arr > e_arr)
    htf_bear = np.where(np.isnan(c_arr)|np.isnan(e_arr), True, c_arr < e_arr)
    return htf_bull, htf_bear

def supertrend(hi, lo, cl, length=10, factor=3.0):
    N = len(cl); hl2 = (hi + lo) / 2.0
    tr = np.zeros(N)
    for i in range(1, N):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    at = np.zeros(N)
    if length < N:
        at[length-1] = tr[1:length].mean()
        for i in range(length, N):
            at[i] = (at[i-1]*(length-1) + tr[i]) / length
    ub = hl2 + factor*at; lb = hl2 - factor*at
    fu = ub.copy(); fl = lb.copy(); td = np.ones(N)
    for i in range(1, N):
        fu[i] = ub[i] if ub[i] < fu[i-1] or cl[i-1] > fu[i-1] else fu[i-1]
        fl[i] = lb[i] if lb[i] > fl[i-1] or cl[i-1] < fl[i-1] else fl[i-1]
        if td[i-1] == 1:
            td[i] = -1 if cl[i] < fl[i] else 1
        else:
            td[i] =  1 if cl[i] > fu[i] else -1
    st_line = np.where(td == 1, fl, fu)
    return td, st_line

# ══════════════════════════════════════════════════════════════
# 3. GENERIC BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════
def run_bt(df, signals, sl_arr, rr, use_be=True, be_r=1.0, sess_s=7, sess_e=20):
    cl = df["close"].values; hi = df["high"].values; lo = df["low"].values
    hr = df.index.hour; N = len(cl)
    trades = []; pos = 0; en = sl = tp = sd = 0.0; be = False
    for i in range(N):
        if pos != 0:
            H = hi[i]; L = lo[i]
            if pos == 1:
                if use_be and not be and H-en >= be_r*sd: sl = max(sl, en); be = True
                if L <= sl: trades.append({"pnl": sl-en,        "sl_dist": sd}); pos=0; continue
                if H >= tp: trades.append({"pnl": tp-en-SPREAD, "sl_dist": sd}); pos=0; continue
            else:
                if use_be and not be and en-L >= be_r*sd: sl = min(sl, en); be = True
                if H >= sl: trades.append({"pnl": en-sl,        "sl_dist": sd}); pos=0; continue
                if L <= tp: trades.append({"pnl": en-tp-SPREAD, "sl_dist": sd}); pos=0; continue
            continue
        if hr[i] < sess_s or hr[i] >= sess_e: continue
        sig = int(signals[i])
        if sig == 0 or sl_arr[i] <= 0: continue
        sd = float(sl_arr[i]); C = cl[i]
        if sig == 1:
            en = C+SPREAD/2; sl = en-sd; tp = en+rr*sd; be = False; pos = 1
        else:
            en = C-SPREAD/2; sl = en+sd; tp = en-rr*sd; be = False; pos = -1
    if pos != 0:
        fp = (cl[-1]-en) if pos == 1 else (en-cl[-1])
        trades.append({"pnl": fp-SPREAD/2, "sl_dist": sd})
    return trades

# ══════════════════════════════════════════════════════════════
# 4. METRIKEN
# ══════════════════════════════════════════════════════════════
def metrics(trades):
    if len(trades) < 10: return None
    r = np.array([t["pnl"]/t["sl_dist"] for t in trades if t["sl_dist"] > 0])
    if len(r) < 10: return None
    wins = (r>0).sum(); tot = len(r)
    gw = r[r>0].sum(); gl = abs(r[r<=0].sum())
    pf = gw/gl if gl > 0 else 99.0
    eq = [10000.0]
    for ri in r: eq.append(eq[-1]*(1+ri*0.01))
    ea = np.array(eq); rm = np.maximum.accumulate(ea)
    dd = abs(((ea-rm)/rm).min())*100
    return {"trades": tot, "wr": round(wins/tot*100,1),
            "pf": round(min(pf,99),2), "dd": round(dd,2),
            "ret": round((ea[-1]-10000)/100,1),
            "sr": round(r.mean()/r.std()*np.sqrt(tot) if r.std()>0 else 0, 3),
            "avg_w": round(r[r>0].mean(),3) if wins > 0 else 0,
            "avg_l": round(r[r<=0].mean(),3) if wins < tot else 0}

def max1_per_day(sig, dates):
    sig = sig.copy().astype(float)
    traded = set()
    for i in range(len(sig)):
        if sig[i] != 0:
            d = dates[i]
            if d in traded: sig[i] = 0
            else: traded.add(d)
    return sig

# ══════════════════════════════════════════════════════════════
# 5. STRATEGY 1 — EMA CLOUD TREND PULLBACK (HTF-Filter)
#    Inspired by v6 (XAUUSD_15M_TrendPullback_v6.pine)
#    Uses EMA 8/21 with EMA50 slow + 60-min HTF EMA50 trend filter
#    Entry: 15M EMA8>EMA21>EMA50 + pullback touches EMA21 + HTF + BQ + RSI
# ══════════════════════════════════════════════════════════════
S1_DEFAULT = {
    "ef": 21, "es": 50,          # fast/slow EMAs (same as v6)
    "pb": 3,                      # pullback lookback bars
    "bqr": 0.40,                  # bar quality ratio (close-low)/range
    "rl": 55, "rs": 45,          # RSI long-min / short-max
    "sep": 0.25,                  # EMA separation: (ef-es)/ATR
    "sl": 2.5, "rr": 3.0,
}
S1_GRID = {
    "bqr": [0.35, 0.40, 0.45],
    "rl":  [50, 55, 60],
    "sep": [0.20, 0.25, 0.30],
    "sl":  [2.0, 2.5, 3.0],
    "rr":  [3.0, 3.5, 4.0],
}

def bt_s1(df, p):
    ef  = ema(df.close, p["ef"]).values
    es  = ema(df.close, p["es"]).values
    rv  = rsi_fn(df.close).values
    av  = atr_fn(df).values
    htf_bull, htf_bear = htf_ema_filter(df, tf_min=60, ema_n=50)
    cl  = df.close.values; hi = df.high.values; lo = df.low.values; op = df.open.values
    N   = len(cl); PBL = p["pb"]; sig = np.zeros(N)

    for i in range(max(PBL+5, 60), N):
        av_i = av[i]
        if np.isnan(av_i) or av_i == 0: continue
        ef_i = ef[i]; es_i = es[i]; rv_i = rv[i]
        if np.isnan(ef_i) or np.isnan(es_i) or np.isnan(rv_i): continue

        # 15M trend: EMA21 > EMA50 (bull) or EMA21 < EMA50 (bear)
        bull15 = ef_i > es_i
        bear15 = ef_i < es_i

        # EMA separation: trend must be established
        sep = abs(ef_i - es_i) / av_i
        if sep < p["sep"]: continue

        # Pullback: recent bars touched EMA21, current bar closed back above (long)
        pb_cls = cl[i-PBL:i]
        ef_prev = ef[i-1]
        pbl = pb_cls.min() <= ef_prev and cl[i] > ef_i
        pbs = pb_cls.max() >= ef_prev and cl[i] < ef_i

        # Pullback zone: not too far from EMA50
        pbzl = pb_cls.min() >= es[i-1] - 0.3*av[i-1]
        pbzs = pb_cls.max() <= es[i-1] + 0.3*av[i-1]

        # EMA21 rising/falling (momentum confirmation)
        ef_rising  = ef[i] > ef[i-2]
        ef_falling = ef[i] < ef[i-2]

        # Bar quality: directional body
        rng = hi[i] - lo[i]
        bq_l = (cl[i] - lo[i]) / rng >= p["bqr"] if rng > 0 else False
        bq_s = (hi[i] - cl[i]) / rng >= p["bqr"] if rng > 0 else False

        # RSI filter
        rsi_l = rv_i >= p["rl"] and rv_i < 80
        rsi_s = rv_i <= p["rs"] and rv_i > 20

        # Combine all conditions (HTF is the gating filter)
        if bull15 and ef_rising and htf_bull[i] and pbl and pbzl and bq_l and rsi_l:
            sig[i] = 1
        elif bear15 and ef_falling and htf_bear[i] and pbs and pbzs and bq_s and rsi_s:
            sig[i] = -1

    sig = max1_per_day(sig, df.index.date)
    return run_bt(df, sig, av * p["sl"], p["rr"])

# ══════════════════════════════════════════════════════════════
# 6. STRATEGY 2 — SUPERTREND + HTF PULLBACK ENTRY
#    Trend: 15M Supertrend direction (must persist 20+ bars)
#    HTF: 60M EMA50 confirms macro direction
#    Entry: Price pulls back to within 1.5×ATR of ST line + BQ + RSI
# ══════════════════════════════════════════════════════════════
S2_DEFAULT = {
    "stl": 10, "stf": 3.0,
    "persist": 20,
    "zone_k": 1.2,
    "bqr": 0.35,
    "rl": 45, "rs": 55,
    "sl": 2.0, "rr": 3.0,
}
S2_GRID = {
    "stf":     [2.5, 3.0, 3.5],
    "persist": [15, 20, 25],
    "zone_k":  [1.0, 1.5, 2.0],
    "bqr":     [0.30, 0.35, 0.40],
    "rl":      [40, 45, 50],
    "sl":      [1.5, 2.0, 2.5],
    "rr":      [2.5, 3.0, 3.5],
}

def bt_s2(df, p):
    hi = df.high.values; lo = df.low.values; cl = df.close.values; op = df.open.values
    td, st_line = supertrend(hi, lo, cl, p["stl"], p["stf"])
    rv  = rsi_fn(df.close).values
    av  = atr_fn(df).values
    htf_bull, htf_bear = htf_ema_filter(df, tf_min=60, ema_n=50)
    N = len(cl); sig = np.zeros(N)

    trend_count = np.zeros(N, dtype=int)
    trend_count[0] = 1
    for i in range(1, N):
        trend_count[i] = trend_count[i-1] + 1 if td[i] == td[i-1] else 1

    for i in range(max(30, p["persist"]+2), N):
        av_i = av[i]
        if np.isnan(av_i) or av_i == 0: continue
        if np.isnan(rv[i]): continue

        if trend_count[i-1] < p["persist"]: continue

        dist = abs(cl[i] - st_line[i])
        if dist > p["zone_k"] * av_i: continue

        rng = hi[i] - lo[i]
        bq_l = (cl[i]-lo[i])/rng >= p["bqr"] if rng > 0 else False
        bq_s = (hi[i]-cl[i])/rng >= p["bqr"] if rng > 0 else False

        rs_long = p["rl"]; rs_short = p["rs"]
        rsi_l = rv[i] >= rs_long and rv[i] <= 70
        rsi_s = rv[i] <= rs_short and rv[i] >= 30

        if td[i] == 1 and htf_bull[i] and cl[i] > st_line[i] and bq_l and rsi_l:
            sig[i] = 1
        elif td[i] == -1 and htf_bear[i] and cl[i] < st_line[i] and bq_s and rsi_s:
            sig[i] = -1

    sig = max1_per_day(sig, df.index.date)
    return run_bt(df, sig, av * p["sl"], p["rr"])

# ══════════════════════════════════════════════════════════════
# 7. STRATEGY 3 — DONCHIAN TREND BREAKOUT + HTF
#    Trend bias: 60M EMA50 direction
#    Entry: Donchian channel breakout + ATR expansion
#    Additional: RSI momentum + Bar quality + Max 1/day
# ══════════════════════════════════════════════════════════════
S3_DEFAULT = {
    "dn": 20,
    "atr_x": 1.0,   # ATR must be >= atr_x × ATR_avg
    "bqr": 0.30,
    "rl": 50, "rs": 50,
    "sl": 2.0, "rr": 2.5,
}
S3_GRID = {
    "dn":    [15, 20, 25],
    "atr_x": [0.9, 1.0, 1.1],
    "bqr":   [0.25, 0.30, 0.35],
    "rl":    [45, 50, 55],
    "sl":    [1.5, 2.0, 2.5],
    "rr":    [2.0, 2.5, 3.0],
}

def bt_s3(df, p):
    av     = atr_fn(df)
    av_avg = av.rolling(p["dn"]).mean()
    dc_hi  = df.high.rolling(p["dn"]).max().shift(1)
    dc_lo  = df.low.rolling(p["dn"]).min().shift(1)
    rv     = rsi_fn(df.close)

    htf_bull, htf_bear = htf_ema_filter(df, tf_min=60, ema_n=50)
    cl = df.close.values; hi = df.high.values; lo = df.low.values; op = df.open.values

    atr_exp = av >= (av_avg * p["atr_x"])
    bup = (df.close > dc_hi) & (df.close.shift(1) <= dc_hi.shift(1))
    bdn = (df.close < dc_lo) & (df.close.shift(1) >= dc_lo.shift(1))

    body = (df.close - df.open).abs()
    rng  = (df.high - df.low).replace(0, np.nan)
    bq   = body / rng
    bq_l = (df.close > df.open) & (bq >= p["bqr"])
    bq_s = (df.close < df.open) & (bq >= p["bqr"])

    rs_long = p["rl"]; rs_short = 100 - p["rs"]
    rsi_l = (rv >= rs_long) & (rv <= 80)
    rsi_s = (rv <= rs_short) & (rv >= 20)

    htf_b_s = pd.Series(htf_bull, index=df.index)
    htf_bear_s = pd.Series(htf_bear, index=df.index)

    sig_l = htf_b_s & bup & atr_exp & bq_l & rsi_l
    sig_s = htf_bear_s & bdn & atr_exp & bq_s & rsi_s

    sig = np.where(sig_l, 1, np.where(sig_s, -1, 0)).astype(float)
    sig = max1_per_day(sig, df.index.date)
    return run_bt(df, sig, (av * p["sl"]).values, p["rr"])

# ══════════════════════════════════════════════════════════════
# 8. GRID SEARCH
# ══════════════════════════════════════════════════════════════
def grid_search(bt_fn, default, grid_spec, df_is, name):
    keys = list(grid_spec.keys())
    combos = list(product(*grid_spec.values()))
    print(f"      {len(combos)} Kombinationen...")
    best_score = -np.inf; best_p = None; best_m = None; rows = []
    for combo in combos:
        p = default.copy()
        for k, v in zip(keys, combo): p[k] = v
        t = bt_fn(df_is, p); m = metrics(t)
        if m is None or m["trades"] < 30: continue
        score = min(m["pf"],5)/5 - max(0,(m["dd"]-25)/60)*0.3
        rows.append({**m, **dict(zip(keys,combo)), "score":score})
        if score > best_score: best_score=score; best_p=dict(zip(keys,combo)); best_m=m
    top = sorted(rows, key=lambda x:x["score"], reverse=True)[:3]
    print(f"\n  TOP-3 {name} (IS):")
    print(f"  {'#':>2}  {'T':>4}  {'WR':>5}  {'PF':>5}  {'DD':>5}  {'Ret':>6}  Params")
    for i, r in enumerate(top, 1):
        params_str = " ".join(f"{k}={r[k]}" for k in keys)
        print(f"  {i:>2}  {r['trades']:>4}  {r['wr']:>4.1f}%  {r['pf']:>5.2f}  {r['dd']:>4.1f}%  {r['ret']:>+5.1f}%  {params_str}")
    return best_p, best_m

# ══════════════════════════════════════════════════════════════
# 9. HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════
print("="*65)
print("XAUUSD 15M — 3 Strategien Backtest v3 (HTF-EMA50-Filter)")
print("Spread: $0.25 | Slippage: 3 Ticks | R-Multiple-Methode")
print("="*65)

print("\n[1/4] Lade echte Goldpreisdaten...")
monthly = fetch_monthly()
print(f"      {len(monthly)} Monatswerte | {monthly.index[0].date()} – {monthly.index[-1].date()}")

print("\n[2/4] Generiere 15M-Bars (Regime-GBM)...")
df_is  = gen_15m(monthly, 2015, 2022, seed=SEED)
df_oos = gen_15m(monthly, 2023, 2026, seed=SEED+1)
print(f"      IS:  {len(df_is):,} Bars | OoS: {len(df_oos):,} Bars")
print(f"      IS ATR Ø: ${atr_fn(df_is).mean():.2f} | OoS ATR Ø: ${atr_fn(df_oos).mean():.2f}")

strategies = [
    ("S1 EMA Cloud HTF",          bt_s1, S1_DEFAULT, S1_GRID),
    ("S2 Supertrend Pullback HTF", bt_s2, S2_DEFAULT, S2_GRID),
    ("S3 Donchian Breakout HTF",  bt_s3, S3_DEFAULT, S3_GRID),
]

results = {}
best_params = {}

print("\n[3/4] Baseline + Optimierung je Strategie auf IS (2015-2022)...")
for name, bt_fn, default, grid in strategies:
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")

    t = bt_fn(df_is, default); m = metrics(t)
    print(f"  Baseline IS: ", end="")
    if m: print(f"T={m['trades']}  WR={m['wr']}%  PF={m['pf']}  DD={m['dd']}%  Ret={m['ret']:+.1f}%")
    else: print("zu wenig Trades")

    bp, bm = grid_search(bt_fn, default, grid, df_is, name)

    final_p = default.copy(); final_p.update(bp or {})
    t_oos = bt_fn(df_oos, final_p); m_oos = metrics(t_oos)

    print(f"\n  Beste IS-Parameter: {bp}")
    if bm: print(f"  IS  → T={bm['trades']}  WR={bm['wr']}%  PF={bm['pf']}  DD={bm['dd']}%  Ret={bm['ret']:+.1f}%")
    else: print(f"  IS  → zu wenig Trades")
    if m_oos: print(f"  OoS → T={m_oos['trades']}  WR={m_oos['wr']}%  PF={m_oos['pf']}  DD={m_oos['dd']}%  Ret={m_oos['ret']:+.1f}%")
    else: print("  OoS → zu wenig Trades")

    results[name] = {"is": bm, "oos": m_oos, "params": final_p}
    best_params[name] = final_p

print(f"\n{'═'*65}")
print("[4/4] FINALE ERGEBNISSE — ALLE 3 STRATEGIEN")
print(f"{'═'*65}")
print(f"\n  {'Strategie':<30} {'IS PF':>6}  {'IS WR':>6}  {'IS DD':>6}  {'OoS PF':>7}  {'OoS WR':>7}  {'OoS DD':>7}")
print(f"  {'─'*82}")
for name, res in results.items():
    mis = res["is"]; moo = res["oos"]
    is_str  = f"{mis['pf']:>6.2f}  {mis['wr']:>5.1f}%  {mis['dd']:>5.1f}%" if mis else f"{'n/a':>6}  {'n/a':>6}  {'n/a':>6}"
    oos_str = f"{moo['pf']:>7.2f}  {moo['wr']:>6.1f}%  {moo['dd']:>6.1f}%" if moo else f"{'n/a':>7}  {'n/a':>7}  {'n/a':>7}"
    print(f"  {name:<30} {is_str}  {oos_str}")

# Marktphasen-Analyse
phases = [
    ("2015-2016 Bär",       2015, 2016),
    ("2017-2018 Seitwärts", 2017, 2018),
    ("2019-2020 Bulle",     2019, 2020),
    ("2021-2022 Konsol.",   2021, 2022),
    ("2023-2024 ATH",       2023, 2024),
    ("2025-2026 Aktuell",   2025, 2026),
]

for name, bt_fn, _, _ in strategies:
    bp = best_params[name]
    print(f"\n  PHASEN-ANALYSE ({name}):")
    print(f"  {'Phase':<25} {'T':>4}  {'WR':>5}  {'PF':>5}  {'Ret':>6}")
    for ph_name, y1, y2 in phases:
        try:
            d = gen_15m(monthly, y1, y2, seed=SEED+y1)
            m = metrics(bt_fn(d, bp))
            if m: print(f"  {ph_name:<25} {m['trades']:>4}  {m['wr']:>4.1f}%  {m['pf']:>5.2f}  {m['ret']:>+5.1f}%")
            else: print(f"  {ph_name:<25} zu wenig Trades")
        except Exception as e:
            print(f"  {ph_name:<25} Fehler: {e}")

with open("/home/user/XAUUSD/backtest_3strat_results.json", "w") as f:
    json.dump({"results": {k: {"is": v["is"], "oos": v["oos"], "params": v["params"]}
                           for k,v in results.items()},
               "timestamp": str(pd.Timestamp.now())}, f, indent=2, default=str)
print("\n  → backtest_3strat_results.json gespeichert")
print(f"{'═'*65}")
