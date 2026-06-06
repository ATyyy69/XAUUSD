"""
ML RSI Strategien — Python Backtest v3 (HTF-Filter)
=====================================================
S1: ML Supertrend Trend Flip + 60M HTF EMA50 Filter
S2: ML RSI Regime Crossover  + 60M HTF EMA50 Filter
S3: ML RSI Trend Pullback    + 60M HTF EMA50 Filter

Zentrale Erkenntnis (analog zu backtest_3strategies.py):
  Der 60M HTF EMA50 Filter ist der entscheidende Faktor.
  Ohne ihn: alle Strategien PF < 1.0 (zufällige GBM-Daten).
  Mit HTF: nur Trades in Makro-Trendrichtung → PF > 1.2 erreichbar.

Approximation der ML RSI Engine:
  stanceState  → RSI-Slope + EMA50-Trend + Chop-Filter
  ML Supertrend → Standard Supertrend(10, factor 3.0–4.0)
  mlRSI        → RSI + gedämpfter Slope-Tilt, EMA(3)

Daten: echte monatliche Goldpreise → synthetische 15M GBM-Bars
IS: 2015-2022 | OOS: 2023-2026
"""
import numpy as np, pandas as pd, json, datetime
import urllib.request, io
from itertools import product as iproduct

SEED   = 42
SPREAD = 0.25

# ══════════════════════════════════════════════════════════════
# 1. DATEN
# ══════════════════════════════════════════════════════════════
def fetch_monthly():
    url = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "py"})
    r = urllib.request.urlopen(req, timeout=20)
    df = pd.read_csv(io.StringIO(r.read().decode()))
    df["Date"] = pd.to_datetime(df["Date"] + "-01")
    df = df.set_index("Date").sort_index()
    df.columns = ["close"]
    return df

def gen_15m(monthly, y1, y2, sigma=0.0013, seed=42):
    MU = 0.000080; WF = 0.55
    rng = np.random.default_rng(seed)
    data = monthly.loc[f"{y1}":f"{y2}"].copy()
    closes, dates = data["close"].values, data.index.tolist()
    oo, hh, ll, cc, ii = [], [], [], [], []
    price = closes[0]
    for m in range(len(dates) - 1):
        lr = np.log(closes[m + 1] / closes[m])
        mu = MU if lr > 0.02 else (-MU if lr < -0.02 else 0.0)
        sm = sigma if abs(lr) > 0.02 else sigma * 0.75
        for day in pd.bdate_range(dates[m], dates[m + 1] - pd.Timedelta(days=1)):
            for b in range(52):
                o = price
                price *= np.exp(mu - 0.5*sm**2 + sm*rng.normal())
                c = price
                h = max(o, c) * (1 + abs(rng.normal(0, sigma * WF)))
                l = min(o, c) * (1 - abs(rng.normal(0, sigma * WF)))
                oo.append(o); hh.append(h); ll.append(l); cc.append(c)
                ii.append(pd.Timestamp(day.date()) +
                          pd.Timedelta(hours=7, minutes=15*b))
    return pd.DataFrame({"open": oo, "high": hh, "low": ll, "close": cc},
                        index=pd.DatetimeIndex(ii))

# ══════════════════════════════════════════════════════════════
# 2. INDIKATOREN
# ══════════════════════════════════════════════════════════════
def ema_s(s, n): return s.ewm(span=n, adjust=False).mean()

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

def htf_filter(df, tf_min=60, ema_n=50):
    """60M HTF EMA50 — forward-filled, 1-bar shift (kein Lookahead)."""
    htf_c = df["close"].resample(f"{tf_min}min").last().dropna()
    htf_e = ema_s(htf_c, ema_n)
    ca = htf_c.reindex(df.index, method="ffill").shift(1).values
    ea = htf_e.reindex(df.index, method="ffill").shift(1).values
    bull = np.where(np.isnan(ca)|np.isnan(ea), False, ca > ea)
    bear = np.where(np.isnan(ca)|np.isnan(ea), False, ca < ea)
    return bull, bear

def ema_np(arr, n):
    """NaN-sicheres numpy EMA."""
    alpha = 2.0 / (n + 1)
    out = arr.copy().astype(float); prev = np.nan
    for i in range(len(out)):
        if np.isnan(arr[i]):
            out[i] = prev if not np.isnan(prev) else np.nan
        elif np.isnan(prev):
            prev = arr[i]; out[i] = arr[i]
        else:
            prev = alpha*arr[i] + (1-alpha)*prev; out[i] = prev
    return out

def supertrend_np(hi, lo, cl, length=10, factor=3.5):
    N = len(cl); hl2 = (hi+lo)/2
    tr = np.zeros(N)
    for i in range(1, N):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    at = np.zeros(N)
    if length < N:
        at[length-1] = tr[1:length].mean()
        for i in range(length, N):
            at[i] = (at[i-1]*(length-1) + tr[i]) / length
    ub = hl2+factor*at; lb = hl2-factor*at
    fu = ub.copy(); fl = lb.copy(); td = np.ones(N, dtype=int)
    for i in range(1, N):
        fu[i] = ub[i] if ub[i]<fu[i-1] or cl[i-1]>fu[i-1] else fu[i-1]
        fl[i] = lb[i] if lb[i]>fl[i-1] or cl[i-1]<fl[i-1] else fl[i-1]
        td[i] = (-1 if cl[i]<fl[i] else 1) if td[i-1]==1 else (1 if cl[i]>fu[i] else -1)
    st_line = np.where(td==1, fl, fu)
    flip_up = np.zeros(N, dtype=bool); flip_dn = np.zeros(N, dtype=bool)
    for i in range(1, N):
        flip_up[i] = td[i]==1  and td[i-1]==-1
        flip_dn[i] = td[i]==-1 and td[i-1]==1
    return td, st_line, flip_up, flip_dn

# ══════════════════════════════════════════════════════════════
# 3. ML INDIKATOREN APPROXIMATION
# ══════════════════════════════════════════════════════════════
def compute_indicators(df, st_factor=3.5, ema_len=50, chop_cut=0.4):
    cl = df.close.values; hi = df.high.values; lo = df.low.values; N = len(cl)
    rsi = rsi_fn(df.close, 14).values
    atr = atr_fn(df, 14).values
    ema50 = ema_s(df.close, ema_len).values
    ema5  = ema_s(df.close, 5).values

    td, st_line, flip_up, flip_dn = supertrend_np(hi, lo, cl, length=10, factor=st_factor)
    htf_bull, htf_bear = htf_filter(df, 60, 50)

    # RSI slope (3-bar, smoothed 5)
    rsi_slope = np.zeros(N)
    for i in range(3, N):
        if not (np.isnan(rsi[i]) or np.isnan(rsi[i-3])):
            rsi_slope[i] = rsi[i] - rsi[i-3]
    rsi_slope = ema_np(rsi_slope, 5)

    # Chop filter
    tf = np.zeros(N)
    for i in range(1, N):
        if atr[i] > 0 and not np.isnan(ema50[i]):
            tf[i] = abs(ema5[i]-ema50[i]) / atr[i]
    chop = tf < chop_cut

    # stanceState (KNN bias proxy)
    stance = np.zeros(N, dtype=int)
    for i in range(60, N):
        if np.isnan(rsi[i]) or np.isnan(ema50[i]): continue
        if rsi_slope[i] > 0.4 and cl[i] > ema50[i] and rsi[i] > 44:
            b = 1
        elif rsi_slope[i] < -0.4 and cl[i] < ema50[i] and rsi[i] < 56:
            b = -1
        else:
            b = 0
        if b == 1 and not chop[i]:    stance[i] = 1
        elif b == -1 and not chop[i]: stance[i] = -1
        else:                          stance[i] = stance[i-1]

    stance_age = np.zeros(N, dtype=int)
    for i in range(1, N):
        stance_age[i] = 0 if stance[i]!=stance[i-1] else stance_age[i-1]+1

    # mlRSI: RSI + slope tilt (±18), EMA(3)
    run_max = np.zeros(N); rm = 1e-6
    for i in range(N):
        v = abs(rsi_slope[i])
        if v > rm: rm = v
        run_max[i] = rm
    tilt = np.where(run_max > 0, np.clip(rsi_slope/run_max, -1, 1)*18, 0)
    raw_ml = np.where(np.isnan(rsi), np.nan, np.clip(rsi+tilt, 0.0, 100.0))
    ml_rsi = ema_np(raw_ml, 3)

    return {'td': td, 'st_line': st_line, 'flip_up': flip_up, 'flip_dn': flip_dn,
            'stance': stance, 'stance_age': stance_age, 'ml_rsi': ml_rsi,
            'rsi': rsi, 'atr': atr, 'chop': chop,
            'htf_bull': htf_bull, 'htf_bear': htf_bear}

# ══════════════════════════════════════════════════════════════
# 4. METRIKEN
# ══════════════════════════════════════════════════════════════
def metrics(trades):
    if len(trades) < 15: return None
    r = np.array([t['pnl']/t['sd'] for t in trades if t['sd']>0 and not np.isnan(t['pnl'])])
    if len(r) < 15: return None
    wins = (r>0).sum(); tot = len(r)
    gw = r[r>0].sum(); gl = abs(r[r<=0].sum())
    pf = gw/gl if gl>0 else 99.0
    eq = [10000.0]
    for ri in r: eq.append(eq[-1]*(1+ri*0.01))
    ea = np.array(eq); rm = np.maximum.accumulate(ea)
    dd = abs(((ea-rm)/rm).min())*100
    sr = r.mean()/r.std()*np.sqrt(tot) if r.std()>0 else 0
    return {'trades': tot, 'wr': round(float(wins/tot*100),1),
            'pf': round(float(min(pf,99)),2), 'dd': round(float(dd),2),
            'ret': round(float((ea[-1]-10000)/100),1),
            'sr': round(float(sr),3),
            'avg_w': round(float(r[r>0].mean()),3) if wins>0 else 0,
            'avg_l': round(float(r[r<=0].mean()),3) if wins<tot else 0}

# ══════════════════════════════════════════════════════════════
# 5. STRATEGIE 1 — ML Supertrend Flip + HTF Filter
#    Entry: ST Flip in HTF-Richtung (nur Flips die mit 60M EMA50 übereinstimmen)
#    SL:    hinter ST-Linie + sl_buf×ATR | TP: rr×SL | Exit: gegenl. Flip
# ══════════════════════════════════════════════════════════════
def bt_s1(df, ind, p):
    cl=df.close.values; hi=df.high.values; lo=df.low.values; hr=df.index.hour; N=len(cl)
    td=ind['td']; st_line=ind['st_line']; flip_up=ind['flip_up']; flip_dn=ind['flip_dn']
    atr=ind['atr']; htf_bull=ind['htf_bull']; htf_bear=ind['htf_bear']
    trades=[]; pos=0; entry=sl=tp=sd=0.0; be=False
    for i in range(60, N):
        if np.isnan(atr[i]) or atr[i]==0: continue
        if hr[i]<7 or hr[i]>=20: continue
        if pos!=0:
            H=hi[i]; L=lo[i]
            if pos==1:
                if p['use_be'] and not be and H-entry>=p['be_r']*sd: sl=max(sl,entry); be=True
                if L<=sl:  trades.append({'pnl':sl-entry,       'sd':sd}); pos=0; continue
                if H>=tp:  trades.append({'pnl':tp-entry-SPREAD,'sd':sd}); pos=0; continue
                if flip_dn[i]: trades.append({'pnl':cl[i]-entry-SPREAD/2,'sd':sd}); pos=0
            else:
                if p['use_be'] and not be and entry-L>=p['be_r']*sd: sl=min(sl,entry); be=True
                if H>=sl:  trades.append({'pnl':entry-sl,       'sd':sd}); pos=0; continue
                if L<=tp:  trades.append({'pnl':entry-tp-SPREAD,'sd':sd}); pos=0; continue
                if flip_up[i]: trades.append({'pnl':entry-cl[i]-SPREAD/2,'sd':sd}); pos=0
            continue
        # Entry: Flip UND HTF-Ausrichtung
        if flip_up[i] and htf_bull[i]:
            sl_r=st_line[i]-p['sl_buf']*atr[i]; sd=max(cl[i]-sl_r, 0.4*atr[i])
            entry=cl[i]+SPREAD/2; sl=entry-sd; tp=entry+p['rr']*sd; be=False; pos=1
        elif flip_dn[i] and htf_bear[i]:
            sl_r=st_line[i]+p['sl_buf']*atr[i]; sd=max(sl_r-cl[i], 0.4*atr[i])
            entry=cl[i]-SPREAD/2; sl=entry+sd; tp=entry-p['rr']*sd; be=False; pos=-1
    return trades

# ══════════════════════════════════════════════════════════════
# 6. STRATEGIE 2 — ML RSI Regime Crossover + HTF Filter
#    Entry: mlRSI kreuzt überzeugend über cross_l (Long) / unter cross_s (Short)
#    Erfordert etabliertes Regime (stance_age >= min_age).
#    SL:    sl_mul×ATR | TP: rr×SL | Exit: Stance dreht gegen Position
# ══════════════════════════════════════════════════════════════
def bt_s2(df, ind, p):
    cl=df.close.values; hi=df.high.values; lo=df.low.values; hr=df.index.hour; N=len(cl)
    stance=ind['stance']; stance_age=ind['stance_age']
    ml_rsi=ind['ml_rsi']; atr=ind['atr']; chop=ind['chop']
    td=ind['td']
    htf_bull=ind['htf_bull']; htf_bear=ind['htf_bear']
    cross_l = p.get('cross_l', 52)   # Long crossover level (default 52)
    cross_s = p.get('cross_s', 48)   # Short crossunder level (default 48)
    trades=[]; pos=0; entry=sl=tp=sd=0.0; be=False
    for i in range(61, N):
        if np.isnan(atr[i]) or atr[i]==0: continue
        if hr[i]<7 or hr[i]>=20: continue
        if np.isnan(ml_rsi[i]) or np.isnan(ml_rsi[i-1]): continue
        if pos!=0:
            H=hi[i]; L=lo[i]
            if pos==1 and stance[i]==-1:
                trades.append({'pnl':cl[i]-entry-SPREAD/2,'sd':sd}); pos=0; continue
            if pos==-1 and stance[i]==1:
                trades.append({'pnl':entry-cl[i]-SPREAD/2,'sd':sd}); pos=0; continue
            if pos==1:
                if p['use_be'] and not be and H-entry>=p['be_r']*sd: sl=max(sl,entry); be=True
                if L<=sl:  trades.append({'pnl':sl-entry,       'sd':sd}); pos=0; continue
                if H>=tp:  trades.append({'pnl':tp-entry-SPREAD,'sd':sd}); pos=0; continue
            else:
                if p['use_be'] and not be and entry-L>=p['be_r']*sd: sl=min(sl,entry); be=True
                if H>=sl:  trades.append({'pnl':entry-sl,       'sd':sd}); pos=0; continue
                if L<=tp:  trades.append({'pnl':entry-tp-SPREAD,'sd':sd}); pos=0; continue
            continue
        if chop[i]: continue
        age_ok = stance_age[i]>=p['min_age']
        # Use configurable crossover levels — 52/48 filters whipsaws vs 50
        cup = ml_rsi[i]>cross_l and ml_rsi[i-1]<=cross_l
        cdn = ml_rsi[i]<cross_s and ml_rsi[i-1]>=cross_s
        # Require ST direction agreement (td==1 for long) — double confirmation
        if stance[i]==1 and td[i]==1 and htf_bull[i] and age_ok and cup:
            sd=max(p['sl_mul']*atr[i], 0.4*atr[i])
            entry=cl[i]+SPREAD/2; sl=entry-sd; tp=entry+p['rr']*sd; be=False; pos=1
        elif stance[i]==-1 and td[i]==-1 and htf_bear[i] and age_ok and cdn:
            sd=max(p['sl_mul']*atr[i], 0.4*atr[i])
            entry=cl[i]-SPREAD/2; sl=entry+sd; tp=entry-p['rr']*sd; be=False; pos=-1
    return trades

# ══════════════════════════════════════════════════════════════
# 7. STRATEGIE 3 — ML RSI Trend Pullback + HTF Filter
#    Entry: etablierter Trend (stance_age≥n + ST + HTF),
#           mlRSI Pullback: bar i-1 unter pb_lo, bar i darüber (Long).
#           Lookahead-freie Variante: crossover aus der neutralen Zone.
#    SL:    hinter ST + sl_buf×ATR | TP: rr×SL | Exit: gegenl. ST Flip
# ══════════════════════════════════════════════════════════════
def bt_s3(df, ind, p):
    cl=df.close.values; hi=df.high.values; lo=df.low.values; hr=df.index.hour; N=len(cl)
    stance=ind['stance']; stance_age=ind['stance_age']
    ml_rsi=ind['ml_rsi']; atr=ind['atr']; chop=ind['chop']
    td=ind['td']; st_line=ind['st_line']
    flip_up=ind['flip_up']; flip_dn=ind['flip_dn']
    htf_bull=ind['htf_bull']; htf_bear=ind['htf_bear']
    trades=[]; pos=0; entry=sl=tp=sd=0.0; be=False
    for i in range(61, N):
        if np.isnan(atr[i]) or atr[i]==0: continue
        if hr[i]<7 or hr[i]>=20: continue
        if np.isnan(ml_rsi[i]) or np.isnan(ml_rsi[i-1]): continue
        if pos!=0:
            H=hi[i]; L=lo[i]
            if pos==1:
                if p['use_be'] and not be and H-entry>=p['be_r']*sd: sl=max(sl,entry); be=True
                if L<=sl:  trades.append({'pnl':sl-entry,       'sd':sd}); pos=0; continue
                if H>=tp:  trades.append({'pnl':tp-entry-SPREAD,'sd':sd}); pos=0; continue
                if flip_dn[i]: trades.append({'pnl':cl[i]-entry-SPREAD/2,'sd':sd}); pos=0
            else:
                if p['use_be'] and not be and entry-L>=p['be_r']*sd: sl=min(sl,entry); be=True
                if H>=sl:  trades.append({'pnl':entry-sl,       'sd':sd}); pos=0; continue
                if L<=tp:  trades.append({'pnl':entry-tp-SPREAD,'sd':sd}); pos=0; continue
                if flip_up[i]: trades.append({'pnl':entry-cl[i]-SPREAD/2,'sd':sd}); pos=0
            continue
        if chop[i]: continue
        age_ok = stance_age[i]>=p['min_age']
        # Simple crossover: bar i-1 war unter pb_lo (pullback), bar i ist drüber (bounce)
        bl = ml_rsi[i-1] <= p['pb_lo'] and ml_rsi[i] > p['pb_lo']
        rs = ml_rsi[i-1] >= p['pb_hi'] and ml_rsi[i] < p['pb_hi']
        if stance[i]==1 and td[i]==1 and htf_bull[i] and age_ok and bl:
            sl_r=st_line[i]-p['sl_buf']*atr[i]; sd=max(cl[i]-sl_r, 0.5*atr[i])
            entry=cl[i]+SPREAD/2; sl=entry-sd; tp=entry+p['rr']*sd; be=False; pos=1
        elif stance[i]==-1 and td[i]==-1 and htf_bear[i] and age_ok and rs:
            sl_r=st_line[i]+p['sl_buf']*atr[i]; sd=max(sl_r-cl[i], 0.5*atr[i])
            entry=cl[i]-SPREAD/2; sl=entry+sd; tp=entry-p['rr']*sd; be=False; pos=-1
    return trades

# ══════════════════════════════════════════════════════════════
# 8. GRID SEARCH (IS-optimiert, Score = PF × Handelsvolumen-Penalty)
# ══════════════════════════════════════════════════════════════
def grid_search(df, ind, bt_fn, grid, min_t=20):
    keys=list(grid.keys()); vals=list(grid.values())
    best={'score':-1,'p':None,'m':None}
    for combo in iproduct(*vals):
        p=dict(zip(keys,combo))
        t=bt_fn(df,ind,p); m=metrics(t)
        if m is None or m['trades']<min_t: continue
        score = m['pf'] * min(1.0, m['trades']/80.0)
        if score > best['score']:
            best={'score':score,'p':dict(p),'m':m}
    return best['p'], best['m']

# ══════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("Lade Golddaten …", flush=True)
    monthly = fetch_monthly()
    print("Generiere 15M-Bars IS (2015-2022) …", flush=True)
    df_is  = gen_15m(monthly, 2015, 2022, seed=SEED)
    print("Generiere 15M-Bars OOS (2023-2026) …", flush=True)
    df_oos = gen_15m(monthly, 2023, 2026, seed=SEED+1)
    print(f"IS: {len(df_is):,} Bars | OOS: {len(df_oos):,} Bars")

    print("\nBerechne ML-Indikatoren (ST factor=3.5, HTF 60M) …", flush=True)
    ind_is  = compute_indicators(df_is,  st_factor=3.5, chop_cut=0.4)
    ind_oos = compute_indicators(df_oos, st_factor=3.5, chop_cut=0.4)

    n_fl  = int(ind_is['flip_up'].sum()+ind_is['flip_dn'].sum())
    n_fh  = int((ind_is['flip_up']&ind_is['htf_bull']).sum()+(ind_is['flip_dn']&ind_is['htf_bear']).sum())
    mlr   = ind_is['ml_rsi']
    n_cr  = int(((mlr[1:]>50)&(mlr[:-1]<=50)).sum()+((mlr[1:]<50)&(mlr[:-1]>=50)).sum())
    print(f"  ST-Flips total={n_fl}  mit HTF-Filter={n_fh}")
    print(f"  mlRSI Crossings 50={n_cr}")
    print(f"  mlRSI range: [{np.nanmin(mlr):.1f}, {np.nanmax(mlr):.1f}]")

    results = {}

    # ── S1 ─────────────────────────────────────────────────────
    print("\n─── S1: ML Supertrend Flip + HTF ───", flush=True)
    p1,m1 = grid_search(df_is, ind_is, bt_s1, {
        'sl_buf': [0.3, 0.5, 0.8],
        'rr':     [2.0, 2.5, 3.0, 3.5],
        'use_be': [True], 'be_r': [0.8, 1.0],
    })
    if p1:
        m1o = metrics(bt_s1(df_oos, ind_oos, p1))
        print(f"  Params: {p1}"); print(f"  IS:  {m1}"); print(f"  OOS: {m1o}")
        results['S1 ML Trend Flip'] = {'is':m1,'oos':m1o,'params':p1}

    # ── S2 ─────────────────────────────────────────────────────
    print("\n─── S2: ML RSI Regime Crossover + ST + HTF ───", flush=True)
    p2,m2 = grid_search(df_is, ind_is, bt_s2, {
        'min_age': [3, 5, 8, 12],
        'cross_l': [50, 52, 54],    # Long crossover level
        'cross_s': [46, 48, 50],    # Short crossunder level
        'sl_mul':  [1.5, 2.0, 2.5],
        'rr':      [1.5, 2.0, 2.5, 3.0],
        'use_be':  [True], 'be_r': [0.6, 0.8, 1.0],
    })
    if p2:
        m2o = metrics(bt_s2(df_oos, ind_oos, p2))
        print(f"  Params: {p2}"); print(f"  IS:  {m2}"); print(f"  OOS: {m2o}")
        results['S2 ML RSI Regime'] = {'is':m2,'oos':m2o,'params':p2}

    # ── S3 ─────────────────────────────────────────────────────
    print("\n─── S3: ML RSI Trend Pullback + ST + HTF ───", flush=True)
    p3,m3 = grid_search(df_is, ind_is, bt_s3, {
        'min_age': [5, 8, 12],
        'pb_lo':   [42, 44, 46, 48],  # Bounce-Trigger (Long)
        'pb_hi':   [52, 54, 56, 58],  # Drop-Trigger (Short)
        'sl_buf':  [0.3, 0.5, 0.8],
        'rr':      [2.0, 2.5, 3.0, 3.5],
        'use_be':  [True], 'be_r': [0.8, 1.0],
    })
    if p3:
        m3o = metrics(bt_s3(df_oos, ind_oos, p3))
        print(f"  Params: {p3}"); print(f"  IS:  {m3}"); print(f"  OOS: {m3o}")
        results['S3 ML Trend Pullback'] = {'is':m3,'oos':m3o,'params':p3}

    # ── ZUSAMMENFASSUNG ────────────────────────────────────────
    print("\n" + "═"*72)
    print("BACKTEST ERGEBNISSE — ML RSI Strategien (XAUUSD 15M, Regime-GBM)")
    print("═"*72)
    print(f"{'Strategie':<26} {'Phase':5} {'Trades':>6} {'WR%':>7} {'PF':>5}"
          f" {'DD%':>6} {'Ret%':>7} {'SR':>7}")
    print("─"*72)
    for name, res in results.items():
        for phase, m in [('IS', res['is']), ('OOS', res['oos'])]:
            if m:
                pf_ok = " ✓" if m['pf']>=1.2 else ""
                wr_ok = " ✓" if m['wr']>=40 else ""
                print(f"{name:<26} {phase:5} {m['trades']:>6} "
                      f"{m['wr']:>6.1f}{wr_ok:<2} {m['pf']:>5.2f}{pf_ok:<2}"
                      f" {m['dd']:>6.2f} {m['ret']:>7.1f} {m['sr']:>7.3f}")
    print("═"*72)
    print("IS=2015-2022 | OOS=2023-2026 | ✓=PF≥1.2 | ✓=WR≥40%")

    out={'results':results,'timestamp':str(datetime.datetime.now())}
    with open('backtest_mlrsi_results.json','w') as f:
        json.dump(out, f, indent=2, default=str)
    print("\nErgebnisse gespeichert → backtest_mlrsi_results.json")

if __name__ == '__main__':
    main()
