# XAUUSD TradingView Pine Script Strategien

Zwei vollständige Pine Script v5 Strategien für Gold (XAUUSD), optimiert für TradingView.

---

## Dateien

| Datei | Timeframe | Typ |
|-------|-----------|-----|
| `XAUUSD_H1_Strategy.pine` | 1 Stunde | Swing / Intraday |
| `XAUUSD_15M_Strategy.pine` | 15 Minuten | Scalping |

---

## H1 Strategie – Übersicht

**Geeignet für:** Intraday- und Swing-Trader  
**Handelszeit:** London + New York Session (07:00–21:00 UTC)

### Verwendete Indikatoren

| Indikator | Parameter | Funktion |
|-----------|-----------|----------|
| EMA 20 | Periode 20 | Schnelle Trendlinie |
| EMA 50 | Periode 50 | Langsame Trendlinie |
| EMA 200 | Periode 200 | Haupttrend-Filter |
| RSI | Periode 14 | Momentum (Long: 45–65, Short: 35–55) |
| MACD | 12/26/9 | Trend-Momentum Bestätigung |
| ATR | Periode 14 | Dynamische SL/TP Berechnung |
| ADX | Periode 14, Min 22 | Trendstärke-Filter |
| Bollinger Bands | 20, 2σ | Volatilitäts-Filter |

### Entry-Bedingungen LONG (alle müssen erfüllt sein)
1. Session aktiv (London oder New York)
2. Preis über EMA 200 (Haupttrend bullish)
3. EMA 20 > EMA 50 (bullishe Ausrichtung)
4. EMA steigt (Steigung über 3 Bars)
5. MACD bullish oder Kreuz über Signallinie
6. RSI zwischen 45–65
7. ADX ≥ 22 (starker Trend)
8. Bollinger Bands nicht zu eng (Mindestvolatilität)

### Risk Management
- **Stop-Loss:** 1.5 × ATR unterhalb Entry
- **Take Profit 1:** 2.0 × ATR (schließt 50% der Position)
- **Take Profit 2:** 4.0 × ATR (schließt restliche 50%)
- **Trailing Stop:** 1.5 × ATR (aktivierbar)
- **Notausstieg:** EMA 20/50 Kreuz gegen die Position

### Risk/Reward
- TP1: 1:1.33 RRR
- TP2: 1:2.67 RRR (Gesamt-RRR ~1:2.0)

---

## 15M Strategie – Übersicht

**Geeignet für:** Scalper und aktive Intraday-Trader  
**Handelszeit:** 07:00–17:00 UTC (London + frühe NY Session)

### Besonderheit: Multi-Timeframe Analyse
Die 15M Strategie zieht den H1-Trend zur Bestätigung heran (`request.security`). Ein Trade gegen den H1-Trend wird blockiert.

### Verwendete Indikatoren

| Indikator | Parameter | Funktion |
|-----------|-----------|----------|
| EMA 9 | Periode 9 | Schnelle Trendlinie |
| EMA 21 | Periode 21 | Langsame Trendlinie |
| EMA 55 | Periode 55 | Lokaler Trend-Filter |
| H1 EMA 20/50 | via HTF | Übergeordneter Trend |
| RSI | Periode 14 | Momentum (Long: >50, Short: <50) |
| MACD | 12/26/9 | Kreuz + Momentum |
| ATR | Periode 14 | SL/TP Berechnung |
| ADX | Periode 14, Min 20 | Trendstärke |
| Bollinger Bands | 20, 2σ | Squeeze-Filter |

### Entry-Bedingungen LONG (alle müssen erfüllt sein)
1. Session aktiv (07:00–17:00 UTC)
2. H1 Trend bullish (H1 EMA 20 > EMA 50)
3. 15M lokaler Trend bullish (Preis > EMA 55, EMA 9 > EMA 21)
4. EMA 9/21 Kreuz nach oben **oder** MACD Kreuz bei bereits bullisher EMA
5. RSI > 50 und nicht überkauft (< 70)
6. MACD Histogram positiv und steigend
7. ADX ≥ 20
8. BB nicht im Squeeze (Mindestvolatilität)
9. Preis über BB-Mittellinie

### Risk Management
- **Stop-Loss:** 1.2 × ATR
- **Take Profit 1:** 1.5 × ATR (schließt 60% der Position)
- **Take Profit 2:** 3.0 × ATR (schließt restliche 40%)
- **Notausstieg:** EMA Kreuz gegen die Position

---

## Installation in TradingView

1. TradingView öffnen → Chart auf XAUUSD setzen
2. Timeframe auf H1 oder 15M stellen
3. Pine Script Editor öffnen (unten: "Pine Editor")
4. Gesamten Inhalt der gewünschten `.pine` Datei einfügen
5. "Zum Chart hinzufügen" klicken
6. Im Strategy Tester Backtest-Ergebnisse prüfen

## Empfohlene Backtest-Einstellungen

- **Zeitraum:** Mindestens 2–3 Jahre historische Daten
- **Startkapital:** 10.000 USD
- **Risiko pro Trade:** 1–2% des Kapitals
- **Spread/Commission:** 0.07 USD (realistisch für Gold CFD)
- **Slippage:** 3 Ticks

## Optimierungs-Hinweise

- ATR-Multiplikatoren je nach Marktvolatilität anpassen
- RSI-Zonen bei seitwärtslaufendem Markt erweitern
- ADX-Schwelle bei trendstarken Phasen erhöhen (25–30)
- Session-Filter deaktivieren für asiatische Session-Tests

## Disclaimer

Diese Strategien dienen ausschließlich zu Bildungszwecken. Kein Handelssystem garantiert Gewinne. Backtest-Ergebnisse spiegeln keine zukünftige Performance wider. Handel mit eigenem Risiko.
