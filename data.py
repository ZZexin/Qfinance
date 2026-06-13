import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ── Market helpers ─────────────────────────────────────────────────────────────

MARKET_EXAMPLES = {
    "🇺🇸 美股": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "META"],
    "🇦🇺 澳股": ["BHP.AX", "CBA.AX", "ANZ.AX", "WBC.AX", "WES.AX", "FMG.AX"],
    "🇨🇳 中国股票": {
        "沪市 A股": ["600519", "601318", "600036", "600900"],   # 茅台/平安/招行/长江电力
        "深市 A股": ["000858", "300750", "002415", "000333"],   # 五粮液/宁德时代/海康/美的
        "港股":     ["0700.HK", "9988.HK", "3690.HK", "2318.HK"],  # 腾讯/阿里/美团/平安
    },
}

CN_EXAMPLE_NAMES = {
    "600519": "贵州茅台", "601318": "中国平安", "600036": "招商银行",
    "600900": "长江电力", "000858": "五粮液", "300750": "宁德时代",
    "002415": "海康威视", "000333": "美的集团",
    "0700.HK": "腾讯控股", "9988.HK": "阿里巴巴",
    "3690.HK": "美团",    "2318.HK": "中国平安H",
}

EXCHANGES = ["沪市 A股", "深市 A股", "港股"]
EXCHANGE_SUFFIX = {"沪市 A股": ".SS", "深市 A股": ".SZ", "港股": ".HK"}


def normalize_ticker(raw: str, market: str, exchange: str = "") -> str:
    """Convert user input to yfinance-compatible ticker."""
    t = raw.strip().upper()
    if market == "🇺🇸 美股":
        return t
    if market == "🇦🇺 澳股":
        return t if t.endswith(".AX") else t + ".AX"
    if market == "🇨🇳 中国股票":
        # Already has a known suffix
        for sfx in (".SS", ".SZ", ".HK"):
            if t.endswith(sfx):
                return t
        base = t.replace(".", "").replace("-", "")
        if exchange in EXCHANGE_SUFFIX:
            sfx = EXCHANGE_SUFFIX[exchange]
            pad = 4 if sfx == ".HK" else 6
            return base.zfill(pad) + sfx
        # Fallback auto-detect
        if base.isdigit():
            if base.startswith("6"):
                return base.zfill(6) + ".SS"
            if base.startswith(("0", "3")) and len(base) == 6:
                return base.zfill(6) + ".SZ"
            return base.zfill(4) + ".HK"
        return t
    return t


def get_currency(ticker: str, market: str) -> str:
    if market == "🇦🇺 澳股":
        return "A$"
    if market == "🇨🇳 中国股票":
        return "HK$" if ticker.endswith(".HK") else "¥"
    return "$"


# ── Technical computation ──────────────────────────────────────────────────────

def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"]

    df["MA20"]  = c.rolling(20).mean()
    df["MA50"]  = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std

    return df


# ── Financial statement helpers ────────────────────────────────────────────────

def extract_row(df: pd.DataFrame, *keywords) -> dict | None:
    if df is None or df.empty:
        return None
    for kw in keywords:
        for idx in df.index:
            if kw.lower() in str(idx).lower():
                row = df.loc[idx]
                result = {}
                for col in row.index:
                    v = row[col]
                    if pd.notna(v):
                        try:
                            result[str(pd.Timestamp(col).year)] = round(float(v) / 1e6, 1)
                        except Exception:
                            pass
                return result or None
    return None


# ── Main fetch ────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str, market: str = "🇺🇸 美股") -> dict:
    """Fetch all data for a yfinance-compatible ticker."""
    stock = yf.Ticker(ticker)

    hist_raw = stock.history(period="1y")
    if hist_raw is None or hist_raw.empty:
        raise ValueError(f"找不到股票代码 '{ticker}'，请确认代码和市场是否正确")

    hist = compute_technicals(hist_raw)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    info = stock.info or {}

    def safe(fn):
        try:
            r = fn()
            return r if r is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    income   = safe(lambda: stock.financials)
    balance  = safe(lambda: stock.balance_sheet)
    cashflow = safe(lambda: stock.cashflow)

    fin = {
        "revenue":          extract_row(income,   "Total Revenue"),
        "gross_profit":     extract_row(income,   "Gross Profit"),
        "operating_income": extract_row(income,   "Operating Income", "EBIT"),
        "net_income":       extract_row(income,   "Net Income"),
        "ebitda":           extract_row(income,   "EBITDA"),
        "total_assets":     extract_row(balance,  "Total Assets"),
        "total_debt":       extract_row(balance,  "Total Debt", "Long Term Debt"),
        "total_equity":     extract_row(balance,  "Stockholders Equity", "Total Equity"),
        "cash":             extract_row(balance,  "Cash And Cash Equivalents", "Cash"),
        "operating_cf":     extract_row(cashflow, "Operating Cash Flow", "Total Cash From Operating"),
        "capex":            extract_row(cashflow, "Capital Expenditure"),
        "free_cf":          extract_row(cashflow, "Free Cash Flow"),
    }

    news = []
    try:
        for a in (stock.news or [])[:8]:
            news.append({
                "title":     a.get("title", ""),
                "publisher": a.get("publisher", ""),
                "date":      datetime.fromtimestamp(a.get("providerPublishTime", 0)).strftime("%Y-%m-%d"),
                "summary":   (a.get("summary") or "")[:180],
                "link":      a.get("link", ""),
            })
    except Exception:
        pass

    earnings_date = None
    try:
        cal = stock.calendar
        if cal is not None and not cal.empty:
            ed = cal.get("Earnings Date")
            if ed is not None and len(ed) > 0:
                earnings_date = str(ed.iloc[0])[:10]
    except Exception:
        pass

    currency = get_currency(ticker, market)

    signals = fetch_signals(ticker)

    return {
        "ticker":        ticker,
        "market":        market,
        "currency":      currency,
        "hist":          hist,
        "info":          info,
        "financials":    fin,
        "news":          news,
        "earnings_date": earnings_date,
        "signals":       signals,
    }


# ── Portfolio price refresh ────────────────────────────────────────────────────

def fetch_current_prices(holdings: list[dict]) -> dict:
    """Returns {ticker: current_price_or_None} for all portfolio holdings."""
    prices = {}
    for h in holdings:
        try:
            hist = yf.Ticker(h["ticker"]).history(period="2d")
            prices[h["ticker"]] = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            prices[h["ticker"]] = None
    return prices


# ── Risk signals ───────────────────────────────────────────────────────────────

def compute_risk_signals(data: dict) -> list[dict]:
    signals = []
    info = data["info"]
    hist = data["hist"]
    cur  = float(hist["Close"].iloc[-1])

    def add(icon, label, level):
        signals.append({"icon": icon, "label": label, "level": level})

    rsi = hist["RSI"].iloc[-1]
    if pd.notna(rsi):
        if rsi > 75:   add("⚠️", f"RSI严重超买 ({rsi:.0f})", "danger")
        elif rsi > 65: add("⚠️", f"RSI偏高/超买 ({rsi:.0f})", "warning")
        elif rsi < 25: add("ℹ️", f"RSI严重超卖 ({rsi:.0f})，关注反弹", "info")
        elif rsi < 35: add("ℹ️", f"RSI偏低/超卖 ({rsi:.0f})", "info")
        else:          add("✅", f"RSI中性 ({rsi:.0f})", "success")

    ma200 = hist["MA200"].iloc[-1]
    ma50  = hist["MA50"].iloc[-1]
    if pd.notna(ma200):
        add("✅" if cur > ma200 else "⚠️",
            f"价格{'高于' if cur > ma200 else '低于'} MA200（长期{'上行' if cur > ma200 else '下行'}趋势）",
            "success" if cur > ma200 else "warning")
    if pd.notna(ma50) and cur < ma50:
        add("⚠️", "价格低于MA50（中期偏弱）", "warning")

    macd_h = hist["MACD_Hist"].iloc[-1]
    if pd.notna(macd_h):
        add("✅" if macd_h > 0 else "⚠️",
            f"MACD柱{'为正（动能向上）' if macd_h > 0 else '为负（动能向下）'}",
            "success" if macd_h > 0 else "warning")

    pe = info.get("trailingPE")
    if pe and isinstance(pe, (int, float)):
        if pe < 0:    add("🔴", f"公司亏损（P/E={pe:.1f}）", "danger")
        elif pe > 60: add("⚠️", f"估值极高 P/E={pe:.1f}，泡沫风险", "danger")
        elif pe > 35: add("⚠️", f"估值偏高 P/E={pe:.1f}", "warning")
        else:         add("✅", f"估值合理 P/E={pe:.1f}", "success")

    de = info.get("debtToEquity")
    if de is not None and isinstance(de, (int, float)):
        if de > 300:   add("🔴", f"负债极高 D/E={de:.0f}%", "danger")
        elif de > 150: add("⚠️", f"负债偏高 D/E={de:.0f}%", "warning")
        else:          add("✅", f"负债可控 D/E={de:.0f}%", "success")

    cr = info.get("currentRatio")
    if cr is not None and isinstance(cr, (int, float)):
        if cr < 1.0:   add("🔴", f"流动比率不足1（{cr:.2f}），短期偿债风险", "danger")
        elif cr < 1.5: add("⚠️", f"流动比率偏低（{cr:.2f}）", "warning")
        else:          add("✅", f"流动性良好（流动比率={cr:.2f}）", "success")

    fcf = info.get("freeCashflow")
    if fcf is not None and isinstance(fcf, (int, float)):
        add("⚠️" if fcf < 0 else "✅",
            f"自由现金流{'为负' if fcf < 0 else '为正'}（{fcf/1e9:.2f}B）",
            "warning" if fcf < 0 else "success")

    rg = info.get("revenueGrowth")
    if rg is not None and isinstance(rg, (int, float)):
        if rg < -0.1:  add("🔴", f"营收大幅下滑（{rg*100:.1f}% YoY）", "danger")
        elif rg < 0:   add("⚠️", f"营收小幅下滑（{rg*100:.1f}% YoY）", "warning")
        elif rg > 0.2: add("✅", f"营收高速增长（+{rg*100:.1f}% YoY）", "success")

    w52h = info.get("fiftyTwoWeekHigh")
    if w52h and isinstance(w52h, (int, float)):
        pct = (cur - w52h) / w52h * 100
        if pct < -40:   add("🔴", f"距52周高点跌幅 {pct:.1f}%，深度回调", "danger")
        elif pct < -20: add("⚠️", f"距52周高点跌幅 {pct:.1f}%", "warning")

    return signals


# ── Multi-source analyst signals ───────────────────────────────────────────────

def fetch_signals(ticker: str) -> dict:
    """
    Fetch multi-source analyst & insider data from yfinance.
    Returns a dict suitable for compute_composite_score().
    All errors are silently swallowed so a missing field never crashes the app.
    """
    stock = yf.Ticker(ticker)
    result: dict = {}

    def safe(fn, transform=None):
        try:
            df = fn()
            if df is None or (hasattr(df, "empty") and df.empty):
                return None
            return transform(df) if transform else df
        except Exception:
            return None

    def _df_to_records(df, n=None):
        df2 = df.copy()
        df2.columns = [str(c) for c in df2.columns]
        for col in df2.columns:
            if pd.api.types.is_datetime64_any_dtype(df2[col]):
                df2[col] = df2[col].dt.strftime("%Y-%m-%d")
        if n:
            df2 = df2.head(n)
        return df2.to_dict(orient="records")

    # ── Analyst recommendation summary (strongBuy/buy/hold/sell/strongSell counts)
    result["rec_summary"] = safe(
        lambda: stock.recommendations_summary,
        lambda df: df.head(4).to_dict(orient="records"),
    )

    # ── Recent upgrades / downgrades (last 6 months)
    def _parse_upgrades(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
        recent = df[df.index >= cutoff].reset_index()
        recent.columns = [str(c) for c in recent.columns]
        date_col = recent.columns[0]
        recent[date_col] = pd.to_datetime(recent[date_col]).dt.strftime("%Y-%m-%d")
        return recent.head(25).to_dict(orient="records")

    result["upgrades_downgrades"] = safe(lambda: stock.upgrades_downgrades, _parse_upgrades)

    # ── Insider transactions
    result["insider_transactions"] = safe(
        lambda: stock.insider_transactions,
        lambda df: _df_to_records(df, n=20),
    )

    # ── Major holders (insider % / institution %)
    result["major_holders"] = safe(
        lambda: stock.major_holders,
        lambda df: df.to_dict(orient="records"),
    )

    # ── Top institutional holders
    result["institutional_holders"] = safe(
        lambda: stock.institutional_holders,
        lambda df: _df_to_records(df, n=10),
    )

    return result


def compute_composite_score(info: dict, signals: dict, hist: pd.DataFrame) -> dict:
    """
    Aggregate five independent signals into a single 0-100 composite score.
    50 = neutral, >57 = bullish, >68 = strongly bullish (mirror image for bearish).

    Weights:
      35% Analyst consensus (strongBuy/buy/hold/sell/strongSell counts)
      20% Price-target upside vs current price
      15% Recent rating upgrades vs downgrades (6-month window)
      15% Insider net buy / sell activity
      15% Technical confirmation (RSI + MACD + MA alignment)
    """

    # ── Component 1: Analyst consensus ────────────────────────────────────────
    rec_list   = signals.get("rec_summary") or []
    latest_rec = rec_list[0] if rec_list else {}
    sb  = int(latest_rec.get("strongBuy",   0) or 0)
    b   = int(latest_rec.get("buy",         0) or 0)
    h   = int(latest_rec.get("hold",        0) or 0)
    s   = int(latest_rec.get("sell",        0) or 0)
    ss  = int(latest_rec.get("strongSell",  0) or 0)
    total_rec  = sb + b + h + s + ss
    n_analysts = int(info.get("numberOfAnalystOpinions") or total_rec or 0)

    consensus    = (sb*100 + b*75 + h*50 + s*25 + ss*0) / total_rec if total_rec else 50.0
    analyst_cred = min(100.0, n_analysts / 25 * 100)

    # ── Component 2: Price-target upside ──────────────────────────────────────
    target = info.get("targetMeanPrice")
    t_low  = info.get("targetLowPrice")
    t_high = info.get("targetHighPrice")
    cur    = float(info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1])
    upside_pct = None

    if target and cur > 0:
        upside     = (target - cur) / cur * 100
        upside_pct = round(upside, 1)
        target_score = max(0.0, min(100.0, 50 + upside * (50 / 30)))   # ±30% → 0–100
        if t_high and t_low and target:
            spread_pct   = (t_high - t_low) / target * 100
            target_cred  = max(0.0, min(100.0, 100 - spread_pct))
        else:
            target_cred  = 40.0
    else:
        target_score = 50.0
        target_cred  = 0.0

    # ── Component 3: Upgrades vs downgrades ───────────────────────────────────
    ud          = signals.get("upgrades_downgrades") or []
    upgrades_n  = sum(1 for r in ud if str(r.get("Action", "")).lower() in ["up",   "upgrade"])
    downgrades_n= sum(1 for r in ud if str(r.get("Action", "")).lower() in ["down", "downgrade"])
    inits_n     = sum(1 for r in ud if str(r.get("Action", "")).lower() in ["init", "reit", "initiated"])
    ud_net      = upgrades_n + downgrades_n
    ud_score    = (upgrades_n / ud_net * 100) if ud_net else 50.0
    ud_cred     = min(100.0, ud_net * 20)

    # ── Component 4: Insider net buy/sell ─────────────────────────────────────
    txns        = signals.get("insider_transactions") or []
    net_val     = 0.0
    buy_count   = sell_count = 0
    for t in txns:
        tx = str(t.get("Transaction", "") or "").lower()
        try:
            val = abs(float(
                str(t.get("Value", 0) or 0)
                .replace(",", "").replace("$", "").replace(" ", "")
            ))
        except (ValueError, TypeError):
            val = 0.0
        if any(w in tx for w in ["purchase", "buy", "acquisition", "acquired"]):
            net_val += val;  buy_count += 1
        elif any(w in tx for w in ["sale", "sell", "sold", "disposed"]):
            net_val -= val;  sell_count += 1

    if net_val > 1e5:
        insider_score = min(100.0, 60 + (net_val / 2e6) * 20)
    elif net_val < -1e5:
        insider_score = max(0.0,  40 + (net_val / 2e6) * 20)
    else:
        insider_score = 50.0
    insider_cred = min(100.0, (buy_count + sell_count) * 15)

    # ── Component 5: Technical confirmation ───────────────────────────────────
    last_rsi   = hist["RSI"].iloc[-1]
    last_macd  = hist["MACD_Hist"].iloc[-1]
    last_ma50  = hist["MA50"].iloc[-1]
    last_ma200 = hist["MA200"].iloc[-1]
    c_price    = float(hist["Close"].iloc[-1])

    rsi_val    = float(last_rsi)  if pd.notna(last_rsi)  else 50.0
    macd_bull  = bool(pd.notna(last_macd)  and float(last_macd)  > 0)
    above_50   = bool(pd.notna(last_ma50)  and c_price > float(last_ma50))
    above_200  = bool(pd.notna(last_ma200) and c_price > float(last_ma200))

    macd_score = 70.0 if macd_bull else 30.0
    if pd.notna(last_ma50) and pd.notna(last_ma200):
        ma_score = 75.0 if (above_50 and above_200) else (25.0 if not (above_50 or above_200) else 50.0)
    else:
        ma_score = 50.0
    tech_score = rsi_val * 0.35 + macd_score * 0.30 + ma_score * 0.35
    tech_cred  = 85.0

    # ── Weighted composite ────────────────────────────────────────────────────
    components = {
        "analyst":   {
            "label": "分析师共识",  "score": consensus,      "weight": 0.35,
            "cred": analyst_cred,
            "counts": {"强买": sb, "买入": b, "持有": h, "卖出": s, "强卖": ss},
            "n_analysts": n_analysts,
        },
        "target":    {
            "label": "目标价空间",  "score": target_score,   "weight": 0.20,
            "cred": target_cred,   "upside_pct": upside_pct,
        },
        "ratings":   {
            "label": "评级动向",    "score": ud_score,        "weight": 0.15,
            "cred": ud_cred,       "upgrades": upgrades_n,
            "downgrades": downgrades_n, "initiations": inits_n,
        },
        "insider":   {
            "label": "内部人交易",  "score": insider_score,   "weight": 0.15,
            "cred": insider_cred,  "net_value": round(net_val),
            "buy_count": buy_count, "sell_count": sell_count,
        },
        "technical": {
            "label": "技术面确认",  "score": tech_score,      "weight": 0.15,
            "cred": tech_cred,     "rsi": round(rsi_val, 1),
            "macd_bull": macd_bull, "above_ma50": above_50, "above_ma200": above_200,
        },
    }

    composite = sum(c["score"] * c["weight"] for c in components.values())
    avg_cred  = sum(c["cred"]  * c["weight"] for c in components.values())

    if   composite >= 68: direction, clr = "强烈看多", "#16a34a"
    elif composite >= 57: direction, clr = "看多",     "#22c55e"
    elif composite >= 43: direction, clr = "中性观望", "#f59e0b"
    elif composite >= 32: direction, clr = "看空",     "#ef4444"
    else:                 direction, clr = "强烈看空", "#dc2626"

    return {
        "composite":   round(composite, 1),
        "credibility": round(avg_cred,  1),
        "direction":   direction,
        "color":       clr,
        "components":  components,
        "n_analysts":  n_analysts,
        "upside_pct":  upside_pct,
        "upgrades":    upgrades_n,
        "downgrades":  downgrades_n,
    }
