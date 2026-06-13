import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf

from data import (
    MARKET_EXAMPLES, EXCHANGES, EXCHANGE_SUFFIX,
    normalize_ticker, get_currency, fetch_stock_data,
    fetch_current_prices, compute_risk_signals,
)
import portfolio as pf

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Qfinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ══════════════════════════════════════════════════════
   CSS Variables — Light Mode (default)
   ══════════════════════════════════════════════════════ */
:root {
  --pill-success-bg:  #d1fae5; --pill-success-bd: #6ee7b7; --pill-success-tx: #065f46;
  --pill-warning-bg:  #fef3c7; --pill-warning-bd: #fcd34d; --pill-warning-tx: #78350f;
  --pill-danger-bg:   #fee2e2; --pill-danger-bd:  #fca5a5; --pill-danger-tx:  #7f1d1d;
  --pill-info-bg:     #dbeafe; --pill-info-bd:    #93c5fd;  --pill-info-tx:    #1e3a8a;
  --news-bg:          #f0f7ff; --news-title-tx:   #111827;  --news-body-tx:    #374151;
  --news-meta-tx:     #6b7280;
}

/* ══════════════════════════════════════════════════════
   CSS Variables — Dark Mode
   Streamlit dark mode follows prefers-color-scheme,
   so this selector covers both the OS setting and the
   Streamlit manual dark toggle (via [data-theme=dark]).
   ══════════════════════════════════════════════════════ */
@media (prefers-color-scheme: dark) { :root {
  --pill-success-bg:  #052e16; --pill-success-bd: #166534; --pill-success-tx: #4ade80;
  --pill-warning-bg:  #3f1a00; --pill-warning-bd: #b45309; --pill-warning-tx: #fcd34d;
  --pill-danger-bg:   #450a0a; --pill-danger-bd:  #991b1b; --pill-danger-tx:  #fca5a5;
  --pill-info-bg:     #0c1a3d; --pill-info-bd:    #1d4ed8;  --pill-info-tx:    #93c5fd;
  --news-bg:          #0f172a; --news-title-tx:   #f1f5f9;  --news-body-tx:    #cbd5e1;
  --news-meta-tx:     #94a3b8;
}}

/* Streamlit's own dark-mode class (manual toggle inside app) */
[data-theme="dark"] {
  --pill-success-bg:  #052e16; --pill-success-bd: #166534; --pill-success-tx: #4ade80;
  --pill-warning-bg:  #3f1a00; --pill-warning-bd: #b45309; --pill-warning-tx: #fcd34d;
  --pill-danger-bg:   #450a0a; --pill-danger-bd:  #991b1b; --pill-danger-tx:  #fca5a5;
  --pill-info-bg:     #0c1a3d; --pill-info-bd:    #1d4ed8;  --pill-info-tx:    #93c5fd;
  --news-bg:          #0f172a; --news-title-tx:   #f1f5f9;  --news-body-tx:    #cbd5e1;
  --news-meta-tx:     #94a3b8;
}

/* ── Global ── */
.main .block-container { padding: 1rem 1rem 2rem; max-width: 900px; }
h1 { font-size: 1.5rem !important; }
h3 { font-size: 1.1rem !important; }

/* ── Risk/signal pills ── */
.pill {
  display: inline-block; border-radius: 6px; padding: 5px 10px;
  margin: 3px 0; font-size: 0.82rem; width: 100%; box-sizing: border-box;
  transition: background 0.2s, color 0.2s;
}
.pill-success { background: var(--pill-success-bg); border: 1px solid var(--pill-success-bd); color: var(--pill-success-tx); }
.pill-warning { background: var(--pill-warning-bg); border: 1px solid var(--pill-warning-bd); color: var(--pill-warning-tx); }
.pill-danger  { background: var(--pill-danger-bg);  border: 1px solid var(--pill-danger-bd);  color: var(--pill-danger-tx);  }
.pill-info    { background: var(--pill-info-bg);    border: 1px solid var(--pill-info-bd);    color: var(--pill-info-tx);    }

/* ── News cards ── */
.news-card {
  border-left: 3px solid #3b82f6;
  background: var(--news-bg);
  border-radius: 4px; padding: 10px 12px; margin-bottom: 8px;
  transition: background 0.2s;
}
.news-card strong { color: var(--news-title-tx); }
.news-card .news-meta { color: var(--news-meta-tx); }
.news-card .news-body { color: var(--news-body-tx); font-size: 0.88rem; }

/* ── Mobile: reduce padding ── */
@media (max-width: 640px) {
  .main .block-container { padding: 0.5rem 0.5rem 2rem; }
  .pill { font-size: 0.78rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

GREEN = "#26a69a"
RED   = "#ef5350"
BLUE  = "#3b82f6"
GRAY  = "#9ca3af"

BASE_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="sans-serif", size=12),
    margin=dict(l=50, r=15, t=40, b=30),
    hovermode="x unified",
)


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_price(v, currency="$"):
    if v is None: return "—"
    return f"{currency}{v:,.2f}"

def fmt_large(v, currency="$"):
    if v is None: return "—"
    if abs(v) >= 1e12: return f"{currency}{v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"{currency}{v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"{currency}{v/1e6:.1f}M"
    return f"{currency}{v:,.0f}"

def fmt_pct(v):
    if v is None: return "—"
    return f"{v*100:.1f}%"

def fmt_num(v, d=2):
    if v is None: return "—"
    try: return f"{float(v):.{d}f}"
    except: return "—"


# ── Chart functions ────────────────────────────────────────────────────────────

def chart_overview(hist: pd.DataFrame, ticker: str, currency: str) -> go.Figure:
    d = hist.iloc[-90:]
    is_up = float(d["Close"].iloc[-1]) >= float(d["Close"].iloc[0])
    line_color = GREEN if is_up else RED
    fill_rgba  = "38,166,154,0.08" if is_up else "239,83,80,0.08"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.02)

    fig.add_trace(go.Scatter(
        x=d.index, y=d["Close"], name="收盘价",
        line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor=f"rgba({fill_rgba})",
        hovertemplate=f"{currency}%{{y:.2f}}<extra>收盘价</extra>",
    ), row=1, col=1)

    for ma, color in [("MA20", "#f59e0b"), ("MA50", BLUE)]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d[ma], name=ma,
            line=dict(color=color, width=1.2, dash="dot"), connectgaps=False,
            hovertemplate=f"{currency}%{{y:.2f}}<extra>{ma}</extra>",
        ), row=1, col=1)

    up = d["Close"] >= d["Open"]
    fig.add_trace(go.Bar(
        x=d.index, y=d["Volume"], name="成交量",
        marker_color=np.where(up, GREEN, RED),
        showlegend=False, opacity=0.65,
    ), row=2, col=1)

    fig.update_layout(**BASE_LAYOUT, height=340,
                      title=dict(text=f"{ticker} — 近90日", x=0.01),
                      legend=dict(orientation="h", y=1.04, x=0),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#f3f4f6", tickprefix=currency)
    fig.update_yaxes(tickprefix="", row=2, col=1)
    return fig


def chart_technical(hist: pd.DataFrame, ticker: str, period: str, currency: str) -> go.Figure:
    n = {"1个月": 30, "3个月": 90, "6个月": 180, "1年": 252}[period]
    d = hist.iloc[-n:]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.48, 0.14, 0.19, 0.19],
                        vertical_spacing=0.025,
                        subplot_titles=("K线 + 均线 + 布林带", "成交量", "RSI (14)", "MACD (12,26,9)"))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="K线",
        increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED),   fillcolor=RED),
    ), row=1, col=1)

    for ma, c in [("MA20", "#f59e0b"), ("MA50", BLUE), ("MA200", "#e11d48")]:
        if d[ma].notna().sum() > 5:
            fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=ma,
                                     line=dict(color=c, width=1.1), connectgaps=False), row=1, col=1)

    if d["BB_Upper"].notna().sum() > 5:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Upper"], name="BB上轨",
                                  line=dict(color=GRAY, width=0.8, dash="dot"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Lower"], name="BB下轨",
                                  line=dict(color=GRAY, width=0.8, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(156,163,175,0.07)", showlegend=False), row=1, col=1)

    # Volume
    up = d["Close"] >= d["Open"]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="成交量",
                          marker_color=np.where(up, GREEN, RED),
                          showlegend=False, opacity=0.7), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI",
                              line=dict(color="#8b5cf6", width=1.5)), row=3, col=1)
    for y, color in [(70, RED), (30, GREEN)]:
        fig.add_hline(y=y, line=dict(color=color, dash="dash", width=0.8), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.04, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.04, row=3, col=1)

    # MACD
    hc = np.where(d["MACD_Hist"] >= 0, GREEN, RED)
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_Hist"],
                          marker_color=hc, showlegend=False, opacity=0.8, name="MACD柱"), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                              line=dict(color=BLUE, width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_Signal"], name="Signal",
                              line=dict(color=RED, width=1.2)), row=4, col=1)

    fig.update_layout(**BASE_LAYOUT, height=680,
                      title=dict(text=f"{ticker} 技术分析 — {period}", x=0.01),
                      legend=dict(orientation="h", y=1.02, x=0),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(gridcolor="#f3f4f6")
    fig.update_yaxes(tickprefix=currency, row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def chart_income(fin: dict) -> go.Figure | None:
    series = {"营收": fin.get("revenue"), "毛利润": fin.get("gross_profit"),
              "营业利润": fin.get("operating_income"), "净利润": fin.get("net_income")}
    series = {k: v for k, v in series.items() if v}
    if not series: return None
    years = sorted({y for v in series.values() for y in v}, reverse=True)[:4]
    colors = [BLUE, GREEN, "#f59e0b", "#8b5cf6"]
    fig = go.Figure()
    for (name, d), color in zip(series.items(), colors):
        vals = [d.get(y) for y in years]
        fig.add_trace(go.Bar(name=name, x=years, y=vals, marker_color=color,
                             text=[f"{v:.0f}" if v else "" for v in vals],
                             textposition="outside", textfont=dict(size=9)))
    fig.update_layout(**BASE_LAYOUT, height=320,
                      title=dict(text="收入与利润（百万）", x=0.01),
                      barmode="group", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(type="category"))
    return fig


def chart_margins(fin: dict) -> go.Figure | None:
    rev = fin.get("revenue") or {}
    gp  = fin.get("gross_profit") or {}
    oi  = fin.get("operating_income") or {}
    ni  = fin.get("net_income") or {}
    years = sorted({y for d in [rev, gp, oi, ni] for y in d}, reverse=True)[:4]
    if not years or not rev: return None
    def mg(num):
        return [round(num[y]/rev[y]*100,1) if y in num and y in rev and rev[y] else None for y in years]
    fig = go.Figure()
    for name, vals, color in [("毛利率", mg(gp), GREEN), ("营业利润率", mg(oi), BLUE), ("净利率", mg(ni), "#8b5cf6")]:
        if any(v is not None for v in vals):
            fig.add_trace(go.Scatter(x=years, y=vals, name=name, mode="lines+markers+text",
                                      line=dict(color=color, width=2), marker=dict(size=6),
                                      text=[f"{v:.1f}%" if v else "" for v in vals],
                                      textposition="top center", textfont=dict(size=9)))
    fig.update_layout(**BASE_LAYOUT, height=280,
                      title=dict(text="利润率趋势（%）", x=0.01),
                      legend=dict(orientation="h", y=1.1, x=0),
                      yaxis=dict(ticksuffix="%", gridcolor="#f3f4f6"),
                      xaxis=dict(type="category"))
    return fig


def chart_cashflow(fin: dict) -> go.Figure | None:
    ocf = fin.get("operating_cf") or {}
    cap = fin.get("capex") or {}
    fcf = fin.get("free_cf") or {}
    years = sorted({y for d in [ocf, cap, fcf] for y in d}, reverse=True)[:4]
    if not years: return None
    fig = go.Figure()
    for name, d, color in [("经营CF", ocf, BLUE), ("资本支出", cap, RED), ("自由CF", fcf, GREEN)]:
        if d:
            vals = [d.get(y) for y in years]
            fig.add_trace(go.Bar(name=name, x=years, y=vals, marker_color=color,
                                  text=[f"{v:.0f}" if v else "" for v in vals],
                                  textposition="outside", textfont=dict(size=9)))
    fig.update_layout(**BASE_LAYOUT, height=280,
                      title=dict(text="现金流（百万）", x=0.01),
                      barmode="group", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(type="category"))
    return fig


def chart_balance(fin: dict) -> go.Figure | None:
    debt   = fin.get("total_debt") or {}
    equity = fin.get("total_equity") or {}
    years  = sorted({y for d in [debt, equity] for y in d}, reverse=True)[:4]
    if not years: return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name="股东权益", x=years, y=[equity.get(y) for y in years],
                          marker_color=GREEN,
                          text=[f"{equity.get(y):.0f}" if equity.get(y) else "" for y in years],
                          textposition="inside", textfont=dict(size=9, color="white")))
    fig.add_trace(go.Bar(name="总负债", x=years, y=[debt.get(y) for y in years],
                          marker_color=RED,
                          text=[f"{debt.get(y):.0f}" if debt.get(y) else "" for y in years],
                          textposition="inside", textfont=dict(size=9, color="white")))
    fig.update_layout(**BASE_LAYOUT, height=280,
                      title=dict(text="资产负债结构（百万）", x=0.01),
                      barmode="stack", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(gridcolor="#f3f4f6"), xaxis=dict(type="category"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("stock_data", None),
    ("market",     "🇺🇸 美股"),
    ("exchange",   "沪市 A股"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## 📊 Qfinance")

# Market selector
st.session_state.market = st.radio(
    "市场", list(MARKET_EXAMPLES.keys()),
    index=list(MARKET_EXAMPLES.keys()).index(st.session_state.market),
    horizontal=True, label_visibility="collapsed",
)
market = st.session_state.market

# CN exchange sub-selector
exchange = ""
if market == "🇨🇳 中国股票":
    st.session_state.exchange = st.radio(
        "交易所", EXCHANGES,
        index=EXCHANGES.index(st.session_state.exchange),
        horizontal=True, label_visibility="collapsed",
    )
    exchange = st.session_state.exchange

# Example tickers
if market == "🇨🇳 中国股票":
    ex_list = MARKET_EXAMPLES[market][exchange]
else:
    ex_list = MARKET_EXAMPLES[market]
st.caption("示例：" + "  ".join(f"`{t}`" for t in ex_list))

# Search bar
c1, c2 = st.columns([5, 1])
with c1:
    raw_ticker = st.text_input(
        "ticker", placeholder={
            "🇺🇸 美股": "输入股票代码，如 AAPL",
            "🇦🇺 澳股": "输入代码，如 BHP（自动加 .AX）",
            "🇨🇳 中国股票": "输入代码数字，如 600519",
        }[market],
        label_visibility="collapsed",
    )
with c2:
    search_btn = st.button("查询 →", use_container_width=True, type="primary")

if search_btn and raw_ticker:
    ticker_norm = normalize_ticker(raw_ticker, market, exchange)
    with st.spinner(f"正在抓取 {ticker_norm} ..."):
        try:
            st.session_state.stock_data = fetch_stock_data(ticker_norm, market)
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"抓取失败：{e}")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

TAB_NAMES = ["📈 概览", "🕯 K线 & 技术", "📑 财务报表", "📰 新闻", "💼 我的持仓"]
tabs = st.tabs(TAB_NAMES)
data = st.session_state.stock_data


# ────────────────────────────────────────────────────────────────────────────
# TABS 1-4: require stock data
# ────────────────────────────────────────────────────────────────────────────

def no_stock():
    st.info("请在上方输入股票代码并点击【查询】")

for i in range(4):
    with tabs[i]:
        if data is None:
            no_stock()

if data:
    info     = data["info"]
    hist     = data["hist"]
    fin      = data["financials"]
    ticker   = data["ticker"]
    currency = data["currency"]
    cur      = float(hist["Close"].iloc[-1])
    prev     = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur
    day_chg  = (cur - prev) / prev * 100

    name = info.get("longName") or info.get("shortName") or ticker

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"**{name}**  `{ticker}`  ·  {info.get('sector','—')} / {info.get('industry','—')}")

        # Row 1: price metrics
        cols = st.columns(5)
        cols[0].metric("当前价格", fmt_price(cur, currency), f"{day_chg:+.2f}%",
                       delta_color="normal")
        cols[1].metric("市值", fmt_large(info.get("marketCap"), currency))
        cols[2].metric("P/E (TTM)", fmt_num(info.get("trailingPE")))
        cols[3].metric("远期 P/E",  fmt_num(info.get("forwardPE")))
        cols[4].metric("EV/EBITDA", fmt_num(info.get("enterpriseToEbitda")))

        # Row 2
        w52h   = info.get("fiftyTwoWeekHigh")
        w52l   = info.get("fiftyTwoWeekLow")
        target = info.get("targetMeanPrice")
        cols2  = st.columns(5)
        cols2[0].metric("52周高点", fmt_price(w52h, currency),
                        f"{(cur-w52h)/w52h*100:+.1f}%" if w52h else None)
        cols2[1].metric("52周低点", fmt_price(w52l, currency),
                        f"{(cur-w52l)/w52l*100:+.1f}%" if w52l else None)
        cols2[2].metric("分析师目标价", fmt_price(target, currency),
                        f"{(target-cur)/cur*100:+.1f}% 空间" if target else None)
        cols2[3].metric("股息率", fmt_pct(info.get("dividendYield")))
        cols2[4].metric("Beta",   fmt_num(info.get("beta")))

        st.plotly_chart(chart_overview(hist, ticker, currency),
                        use_container_width=True, config={"displayModeBar": False})

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### ⚠️ 风险与信号")
            for s in compute_risk_signals(data):
                st.markdown(
                    f'<div class="pill pill-{s["level"]}">{s["icon"]} {s["label"]}</div>',
                    unsafe_allow_html=True)

        with col_r:
            st.markdown("#### 📋 基本面速览")
            ratio_rows = [
                ("毛利率",        fmt_pct(info.get("grossMargins"))),
                ("营业利润率",    fmt_pct(info.get("operatingMargins"))),
                ("净利率",        fmt_pct(info.get("profitMargins"))),
                ("ROE",          fmt_pct(info.get("returnOnEquity"))),
                ("ROA",          fmt_pct(info.get("returnOnAssets"))),
                ("营收增长 YoY", fmt_pct(info.get("revenueGrowth"))),
                ("净利增长 YoY", fmt_pct(info.get("earningsGrowth"))),
                ("流动比率",      fmt_num(info.get("currentRatio"))),
                ("负债权益比",    fmt_num(info.get("debtToEquity"), 1)),
                ("自由现金流",    fmt_large(info.get("freeCashflow"), currency)),
            ]
            st.dataframe(pd.DataFrame(ratio_rows, columns=["指标", "数值"]),
                         hide_index=True, use_container_width=True, height=360)

        summary = info.get("longBusinessSummary")
        if summary:
            with st.expander("公司简介"):
                st.write(summary)

    # ── Tab 2: Technical ──────────────────────────────────────────────────────
    with tabs[1]:
        period = st.radio("周期", ["1个月", "3个月", "6个月", "1年"],
                          index=2, horizontal=True, key="period_radio")
        st.plotly_chart(chart_technical(hist, ticker, period, currency),
                        use_container_width=True,
                        config={"displayModeBar": True, "displaylogo": False})

        # Quick reference table
        rsi_v   = hist["RSI"].iloc[-1]
        macd_h  = hist["MACD_Hist"].iloc[-1]
        bb_up   = hist["BB_Upper"].iloc[-1]
        bb_lo   = hist["BB_Lower"].iloc[-1]

        tech_rows = [
            ("当前价格",      fmt_price(cur, currency)),
            ("MA20",          fmt_price(hist["MA20"].iloc[-1], currency) if pd.notna(hist["MA20"].iloc[-1]) else "—"),
            ("MA50",          fmt_price(hist["MA50"].iloc[-1], currency) if pd.notna(hist["MA50"].iloc[-1]) else "—"),
            ("MA200",         fmt_price(hist["MA200"].iloc[-1], currency) if pd.notna(hist["MA200"].iloc[-1]) else "—"),
            ("价格 vs MA50",  ("↑ 高于" if cur > hist["MA50"].iloc[-1] else "↓ 低于") if pd.notna(hist["MA50"].iloc[-1]) else "—"),
            ("价格 vs MA200", ("↑ 高于" if cur > hist["MA200"].iloc[-1] else "↓ 低于") if pd.notna(hist["MA200"].iloc[-1]) else "—"),
            ("RSI (14)",      f"{rsi_v:.1f}  {'超买⚠️' if rsi_v>70 else '超卖ℹ️' if rsi_v<30 else '中性✅'}" if pd.notna(rsi_v) else "—"),
            ("MACD 柱",       f"{macd_h:.4f}  {'看涨✅' if macd_h>0 else '看跌⚠️'}" if pd.notna(macd_h) else "—"),
            ("布林上轨",      fmt_price(bb_up, currency) if pd.notna(bb_up) else "—"),
            ("布林下轨",      fmt_price(bb_lo, currency) if pd.notna(bb_lo) else "—"),
        ]
        st.dataframe(pd.DataFrame(tech_rows, columns=["指标", "数值"]),
                     hide_index=True, use_container_width=True)

    # ── Tab 3: Financials ─────────────────────────────────────────────────────
    with tabs[2]:
        fin_currency_note = {"🇺🇸 美股": "USD", "🇦🇺 澳股": "AUD", "🇨🇳 中国股票": "CNY/HKD"}
        st.caption(f"单位：百万 {fin_currency_note.get(market, '')}")

        c1, c2 = st.columns(2)
        with c1:
            f = chart_income(fin)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
            else: st.info("暂无收入数据")
        with c2:
            f = chart_margins(fin)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
            else: st.info("暂无利润率数据")

        c3, c4 = st.columns(2)
        with c3:
            f = chart_cashflow(fin)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
            else: st.info("暂无现金流数据")
        with c4:
            f = chart_balance(fin)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
            else: st.info("暂无负债数据")

        # Raw table
        all_years = sorted({y for v in fin.values() if v for y in v}, reverse=True)[:4]
        if all_years:
            labels = [("revenue","总营收"),("gross_profit","毛利润"),("operating_income","营业利润"),
                      ("net_income","净利润"),("ebitda","EBITDA"),("operating_cf","经营现金流"),
                      ("capex","资本支出"),("free_cf","自由现金流"),("total_assets","总资产"),
                      ("total_debt","总负债"),("total_equity","股东权益"),("cash","现金")]
            rows = []
            for key, label in labels:
                d = fin.get(key) or {}
                row = {"项目": label}
                for y in all_years:
                    row[y] = f"{d[y]:,.1f}" if y in d and d[y] is not None else "—"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Tab 4: News ───────────────────────────────────────────────────────────
    with tabs[3]:
        if data.get("earnings_date"):
            st.info(f"📅 下次财报日期：**{data['earnings_date']}**")

        rec     = info.get("recommendationKey", "").upper()
        n_an    = info.get("numberOfAnalystOpinions")
        t_mean  = info.get("targetMeanPrice")
        t_high  = info.get("targetHighPrice")
        t_low   = info.get("targetLowPrice")

        if rec or t_mean:
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("分析师评级",   rec or "—")
            ac2.metric("覆盖人数",      str(n_an) if n_an else "—")
            ac3.metric("目标价（均）",  fmt_price(t_mean, currency),
                       f"{(t_mean-cur)/cur*100:+.1f}%" if t_mean else None)
            ac4.metric("目标价区间",
                       f"{currency}{t_low:.0f}–{currency}{t_high:.0f}" if t_low and t_high else "—")

        st.markdown("")
        news = data.get("news", [])
        if news:
            st.markdown(f"#### 最新新闻（{len(news)} 条）")
            for item in news:
                st.markdown(
                    f'<div class="news-card">'
                    f'<strong>{item["title"]}</strong><br>'
                    f'<span class="news-meta">{item["publisher"]} · {item["date"]}</span>'
                    + (f'<br><span class="news-body">{item["summary"]}</span>' if item.get("summary") else "")
                    + "</div>",
                    unsafe_allow_html=True)
        else:
            st.info("暂无新闻数据")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5: PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    holdings = pf.load()

    # ── Add / Edit holding ────────────────────────────────────────────────────
    with st.expander("➕ 添加 / 更新持仓", expanded=(len(holdings) == 0)):
        fc1, fc2 = st.columns(2)
        with fc1:
            p_market = st.selectbox("市场", list(MARKET_EXAMPLES.keys()), key="p_market")
            p_exchange = ""
            if p_market == "🇨🇳 中国股票":
                p_exchange = st.selectbox("交易所", EXCHANGES, key="p_exchange")
            p_raw = st.text_input("股票代码", placeholder="如 AAPL / BHP / 600519", key="p_ticker")
            p_currency = get_currency(
                normalize_ticker(p_raw, p_market, p_exchange) if p_raw else "",
                p_market
            )
        with fc2:
            p_shares = st.number_input("持仓股数", min_value=0.0001, value=100.0,
                                        format="%.4f", key="p_shares")
            p_cost   = st.number_input(f"成本价（{p_currency}）",
                                        min_value=0.0001, value=100.0,
                                        format="%.4f", key="p_cost")
            p_name   = st.text_input("备注名称（可选）", key="p_name")

        if st.button("💾 保存", use_container_width=True, type="primary", key="p_save"):
            if not p_raw:
                st.warning("请输入股票代码")
            else:
                norm = normalize_ticker(p_raw, p_market, p_exchange)
                # Try to get company name
                name_auto = p_name
                if not name_auto:
                    try:
                        i = yf.Ticker(norm).info
                        name_auto = i.get("shortName", "") or i.get("longName", "")
                    except Exception:
                        pass
                action = pf.upsert(norm, p_market, p_exchange, p_shares, p_cost,
                                   name_auto, p_currency)
                st.success(f"{'✅ 已添加' if action == 'added' else '🔄 已更新'} {norm}")
                st.rerun()

    st.markdown("")

    # ── Holdings display ──────────────────────────────────────────────────────
    holdings = pf.load()

    if not holdings:
        st.info("尚无持仓记录。点击上方「添加持仓」开始记录。")
    else:
        # Fetch current prices
        with st.spinner("刷新实时价格..."):
            prices = fetch_current_prices(holdings)

        # Build display rows
        rows = []
        total_cost_usd  = 0.0   # rough sum (mixed currency, just for visual)
        total_value_usd = 0.0

        for h in holdings:
            t      = h["ticker"]
            cur_p  = prices.get(t)
            cost_p = h["cost_price"]
            shares = h["shares"]
            ccy    = h.get("currency", "$")

            cost_val = shares * cost_p
            cur_val  = shares * cur_p if cur_p else None
            pnl      = (cur_val - cost_val) if cur_val is not None else None
            pnl_pct  = (pnl / cost_val * 100) if pnl is not None and cost_val else None

            rows.append({
                "代码":      t,
                "市场":      h.get("market", "—").split()[0],   # just the flag
                "名称":      h.get("name", "—")[:12],
                "股数":      shares,
                "成本价":    f"{ccy}{cost_p:,.2f}",
                "现价":      f"{ccy}{cur_p:,.2f}" if cur_p else "—",
                "市值":      f"{ccy}{cur_val:,.0f}" if cur_val else "—",
                "盈亏":      f"{ccy}{pnl:+,.0f}" if pnl is not None else "—",
                "盈亏%":     f"{pnl_pct:+.1f}%" if pnl_pct is not None else "—",
                "买入日":    h.get("added", "—"),
                "_pnl_val":  pnl,
                "_ticker":   t,
                "_market":   h.get("market",""),
            })

        # Summary metrics
        total_cost  = sum(h["shares"] * h["cost_price"] for h in holdings)
        total_value = sum(h["shares"] * prices.get(h["ticker"], h["cost_price"]) for h in holdings)
        total_pnl   = total_value - total_cost
        total_pnl_p = total_pnl / total_cost * 100 if total_cost else 0

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("总持仓数", f"{len(holdings)} 只")
        mc2.metric("总成本", f"{total_cost:,.0f}")
        mc3.metric("当前市值", f"{total_value:,.0f}", f"{total_pnl:+,.0f}")
        mc4.metric("总盈亏%", f"{total_pnl_p:+.1f}%",
                   delta_color="normal" if total_pnl_p >= 0 else "inverse")

        st.markdown("")

        # Holdings table (display-only rows)
        display_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        st.dataframe(
            pd.DataFrame(display_rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "盈亏":  st.column_config.TextColumn("盈亏"),
                "盈亏%": st.column_config.TextColumn("盈亏%"),
            },
            height=min(40 * len(rows) + 55, 400),
        )

        # Delete holdings
        st.markdown("#### 删除持仓")
        del_cols = st.columns(min(len(holdings), 4))
        for i, h in enumerate(holdings):
            col = del_cols[i % len(del_cols)]
            label = f"🗑 {h['ticker']}"
            if col.button(label, key=f"del_{h['ticker']}_{h['market']}", use_container_width=True):
                pf.remove(h["ticker"], h["market"])
                st.success(f"已删除 {h['ticker']}")
                st.rerun()

