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

    return {
        "ticker":        ticker,
        "market":        market,
        "currency":      currency,
        "hist":          hist,
        "info":          info,
        "financials":    fin,
        "news":          news,
        "earnings_date": earnings_date,
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
