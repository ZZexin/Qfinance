import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data import fetch_stock_data, compute_risk_signals

st.set_page_config(
    page_title="Qfinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
  .metric-card {
      background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px;
      padding: 12px 16px; text-align: center;
  }
  .metric-val { font-size: 1.4rem; font-weight: 700; margin: 4px 0; }
  .metric-lbl { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
  .signal-success { background:#d1fae5; border:1px solid #6ee7b7; border-radius:6px; padding:6px 12px; margin:3px 0; font-size:0.85rem; }
  .signal-warning { background:#fef3c7; border:1px solid #fcd34d; border-radius:6px; padding:6px 12px; margin:3px 0; font-size:0.85rem; }
  .signal-danger  { background:#fee2e2; border:1px solid #fca5a5; border-radius:6px; padding:6px 12px; margin:3px 0; font-size:0.85rem; }
  .signal-info    { background:#dbeafe; border:1px solid #93c5fd; border-radius:6px; padding:6px 12px; margin:3px 0; font-size:0.85rem; }
  .news-card { border-left: 3px solid #3b82f6; background: #f8faff; border-radius: 4px; padding: 10px 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

GREEN = "#26a69a"
RED   = "#ef5350"
BLUE  = "#3b82f6"
GRAY  = "#9ca3af"

PLOT_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="sans-serif", size=12),
    margin=dict(l=50, r=20, t=40, b=30),
    hovermode="x unified",
)


def fmt_large(v):
    if v is None:
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


def fmt_num(v, decimals=2):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


# ── chart builders ────────────────────────────────────────────────────────────

def chart_overview(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """90-day price line + MA20/MA50 + volume subplot."""
    d = hist.iloc[-90:]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )

    color = GREEN if float(d["Close"].iloc[-1]) >= float(d["Close"].iloc[0]) else RED
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Close"],
        name="收盘价", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({'38,166,154' if color == GREEN else '239,83,80'},0.06)",
    ), row=1, col=1)

    for ma, c in [("MA20", "#f59e0b"), ("MA50", BLUE)]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d[ma], name=ma,
            line=dict(color=c, width=1.3, dash="dot"),
        ), row=1, col=1)

    up = d["Close"] >= d["Open"]
    fig.add_trace(go.Bar(
        x=d.index, y=d["Volume"], name="成交量",
        marker_color=np.where(up, GREEN, RED),
        showlegend=False, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        height=380,
        title=dict(text=f"{ticker} — 近90日走势", x=0.01),
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="#f3f4f6")
    fig.update_xaxes(gridcolor="#f3f4f6")
    return fig


def chart_technical(hist: pd.DataFrame, ticker: str, period: str) -> go.Figure:
    """Candlestick + MA + Volume + RSI + MACD (4 panes)."""
    days_map = {"1个月": 30, "3个月": 90, "6个月": 180, "1年": 252}
    d = hist.iloc[-days_map.get(period, 252):]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.14, 0.18, 0.18],
        vertical_spacing=0.025,
        subplot_titles=("K线 + 均线", "成交量", "RSI (14)", "MACD (12, 26, 9)"),
    )

    # ── Candlestick
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="K线",
        increasing=dict(line=dict(color=GREEN), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED),   fillcolor=RED),
    ), row=1, col=1)

    for ma, c, w in [("MA20", "#f59e0b", 1.2), ("MA50", BLUE, 1.2), ("MA200", RED, 1.2)]:
        if d[ma].notna().sum() > 5:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[ma], name=ma,
                line=dict(color=c, width=w), connectgaps=False,
            ), row=1, col=1)

    # Bollinger Bands
    if d["BB_Upper"].notna().sum() > 5:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_Upper"], name="BB上轨",
            line=dict(color=GRAY, width=0.8, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_Lower"], name="BB下轨",
            line=dict(color=GRAY, width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(156,163,175,0.07)",
            showlegend=False,
        ), row=1, col=1)

    # ── Volume
    up = d["Close"] >= d["Open"]
    fig.add_trace(go.Bar(
        x=d.index, y=d["Volume"],
        marker_color=np.where(up, GREEN, RED),
        showlegend=False, opacity=0.75, name="成交量",
    ), row=2, col=1)

    # ── RSI
    fig.add_trace(go.Scatter(
        x=d.index, y=d["RSI"], name="RSI",
        line=dict(color="#8b5cf6", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=RED,   dash="dash", width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=GREEN, dash="dash", width=0.8), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED,   opacity=0.04, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.04, row=3, col=1)

    # ── MACD
    hist_colors = np.where(d["MACD_Hist"] >= 0, GREEN, RED)
    fig.add_trace(go.Bar(
        x=d.index, y=d["MACD_Hist"],
        marker_color=hist_colors, showlegend=False, opacity=0.8, name="MACD柱",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD"], name="MACD",
        line=dict(color=BLUE, width=1.2),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD_Signal"], name="Signal",
        line=dict(color=RED, width=1.2),
    ), row=4, col=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        height=800,
        title=dict(text=f"{ticker} 技术分析", x=0.01),
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="#f3f4f6")
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_xaxes(gridcolor="#f3f4f6")
    return fig


def chart_income(fin: dict) -> go.Figure | None:
    """Grouped bar: Revenue / Gross Profit / Operating Income / Net Income."""
    series = {
        "营收":     fin.get("revenue"),
        "毛利润":   fin.get("gross_profit"),
        "营业利润": fin.get("operating_income"),
        "净利润":   fin.get("net_income"),
    }
    series = {k: v for k, v in series.items() if v}
    if not series:
        return None

    years = sorted({y for v in series.values() for y in v.keys()}, reverse=True)[:4]
    colors = [BLUE, GREEN, "#f59e0b", "#8b5cf6"]

    fig = go.Figure()
    for (name, data), color in zip(series.items(), colors):
        fig.add_trace(go.Bar(
            name=name,
            x=years,
            y=[data.get(y) for y in years],
            marker_color=color,
            text=[f"${data.get(y):.0f}M" if data.get(y) is not None else "" for y in years],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=380,
        title=dict(text="收入与利润趋势（百万美元）", x=0.01),
        barmode="group",
        legend=dict(orientation="h", y=1.06, x=0),
        yaxis=dict(gridcolor="#f3f4f6"),
        xaxis=dict(type="category"),
    )
    return fig


def chart_margins(fin: dict) -> go.Figure | None:
    """Line chart: gross / operating / net margins computed from financials."""
    rev  = fin.get("revenue") or {}
    gp   = fin.get("gross_profit") or {}
    oi   = fin.get("operating_income") or {}
    ni   = fin.get("net_income") or {}

    years = sorted({y for d in [rev, gp, oi, ni] for y in d.keys()}, reverse=True)[:4]
    if not years or not rev:
        return None

    def margin(num_dict):
        return [
            round(num_dict[y] / rev[y] * 100, 1)
            if y in num_dict and y in rev and rev[y] and rev[y] != 0
            else None
            for y in years
        ]

    gm = margin(gp)
    om = margin(oi)
    nm = margin(ni)

    fig = go.Figure()
    for name, vals, color in [("毛利率", gm, GREEN), ("营业利润率", om, BLUE), ("净利率", nm, "#8b5cf6")]:
        if any(v is not None for v in vals):
            fig.add_trace(go.Scatter(
                x=years, y=vals, name=name,
                mode="lines+markers+text",
                line=dict(color=color, width=2),
                marker=dict(size=7),
                text=[f"{v:.1f}%" if v is not None else "" for v in vals],
                textposition="top center",
                textfont=dict(size=10),
            ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        title=dict(text="利润率趋势（%）", x=0.01),
        legend=dict(orientation="h", y=1.1, x=0),
        yaxis=dict(ticksuffix="%", gridcolor="#f3f4f6"),
        xaxis=dict(type="category"),
    )
    return fig


def chart_cashflow(fin: dict) -> go.Figure | None:
    """Bar chart: Operating CF / Capex / Free CF."""
    ocf  = fin.get("operating_cf") or {}
    cap  = fin.get("capex") or {}
    fcf  = fin.get("free_cf") or {}

    years = sorted({y for d in [ocf, cap, fcf] for y in d.keys()}, reverse=True)[:4]
    if not years:
        return None

    fig = go.Figure()
    for name, data, color in [
        ("经营现金流", ocf, BLUE),
        ("资本支出",   cap, RED),
        ("自由现金流", fcf, GREEN),
    ]:
        if data:
            vals = [data.get(y) for y in years]
            fig.add_trace(go.Bar(
                name=name, x=years, y=vals,
                marker_color=color,
                text=[f"${v:.0f}M" if v is not None else "" for v in vals],
                textposition="outside",
                textfont=dict(size=10),
            ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=340,
        title=dict(text="现金流趋势（百万美元）", x=0.01),
        barmode="group",
        legend=dict(orientation="h", y=1.06, x=0),
        yaxis=dict(gridcolor="#f3f4f6"),
        xaxis=dict(type="category"),
    )
    return fig


def chart_balance(fin: dict) -> go.Figure | None:
    """Stacked bar: equity vs debt."""
    debt   = fin.get("total_debt")   or {}
    equity = fin.get("total_equity") or {}

    years = sorted({y for d in [debt, equity] for y in d.keys()}, reverse=True)[:4]
    if not years:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="股东权益", x=years,
        y=[equity.get(y) for y in years],
        marker_color=GREEN,
        text=[f"${equity.get(y):.0f}M" if equity.get(y) else "" for y in years],
        textposition="inside", textfont=dict(size=10, color="white"),
    ))
    fig.add_trace(go.Bar(
        name="总负债", x=years,
        y=[debt.get(y) for y in years],
        marker_color=RED,
        text=[f"${debt.get(y):.0f}M" if debt.get(y) else "" for y in years],
        textposition="inside", textfont=dict(size=10, color="white"),
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=320,
        title=dict(text="资产负债结构（百万美元）", x=0.01),
        barmode="stack",
        legend=dict(orientation="h", y=1.06, x=0),
        yaxis=dict(gridcolor="#f3f4f6"),
        xaxis=dict(type="category"),
    )
    return fig


# ── main UI ───────────────────────────────────────────────────────────────────

st.markdown("## 📊 Qfinance · 股票数据仪表板")
st.caption("输入股票代码，自动抓取价格、财报、技术指标、新闻，图表化展示")

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    ticker_input = st.text_input(
        "ticker",
        placeholder="输入股票代码，如 AAPL、TSLA、BABA、0700.HK",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("查询 →", use_container_width=True, type="primary")

st.markdown(
    "示例：" + "  ".join([f"`{t}`" for t in ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "0700.HK", "BABA"]]),
    unsafe_allow_html=False,
)

st.divider()

# Session state: persist data across tab switches
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if search_btn and ticker_input:
    with st.spinner(f"正在抓取 {ticker_input.upper()} 数据..."):
        try:
            st.session_state.stock_data = fetch_stock_data(ticker_input)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"抓取数据时出错：{e}")
            st.stop()

data = st.session_state.stock_data
if data is None:
    st.markdown("""
| 板块 | 内容 |
|------|------|
| 📈 概览 | 关键指标、近90日走势、风险信号 |
| 🕯 K线&技术指标 | 蜡烛图、MA/布林带、RSI、MACD、成交量 |
| 📑 财务报表 | 收入利润趋势、利润率、现金流、资产负债 |
| 📰 新闻&事件 | 最新公告、新闻、下次财报日期 |
""")
    st.stop()

info      = data["info"]
hist      = data["hist"]
fin       = data["financials"]
ticker    = data["ticker"]
cur_price = float(hist["Close"].iloc[-1])
prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur_price
day_chg   = (cur_price - prev_close) / prev_close * 100

company_name = info.get("longName") or info.get("shortName") or ticker

st.markdown(f"### {company_name}（{ticker}）")
st.caption(f"{info.get('sector','—')} · {info.get('industry','—')} · {info.get('country','—')}")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📈 概览", "🕯 K线 & 技术指标", "📑 财务报表", "📰 新闻 & 事件"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:

    # ── Metric cards row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    delta_color = "normal"
    c1.metric("当前价格", f"${cur_price:.2f}", f"{day_chg:+.2f}%", delta_color=delta_color)
    c2.metric("市值",      fmt_large(info.get("marketCap")))
    c3.metric("P/E（TTM）", fmt_num(info.get("trailingPE")))
    c4.metric("远期P/E",   fmt_num(info.get("forwardPE")))
    c5.metric("EV/EBITDA", fmt_num(info.get("enterpriseToEbitda")))

    # ── Metric cards row 2
    w52h = info.get("fiftyTwoWeekHigh")
    w52l = info.get("fiftyTwoWeekLow")
    target = info.get("targetMeanPrice")
    upside = ((target - cur_price) / cur_price * 100) if target else None

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("52周高点", f"${w52h:.2f}" if w52h else "N/A",
              f"{(cur_price-w52h)/w52h*100:+.1f}%" if w52h else None)
    c7.metric("52周低点", f"${w52l:.2f}" if w52l else "N/A",
              f"{(cur_price-w52l)/w52l*100:+.1f}%" if w52l else None)
    c8.metric("分析师目标价", f"${target:.2f}" if target else "N/A",
              f"潜在空间 {upside:+.1f}%" if upside else None)
    c9.metric("股息率",  fmt_pct(info.get("dividendYield")))
    c10.metric("Beta",   fmt_num(info.get("beta")))

    st.markdown("")

    # ── Overview chart
    st.plotly_chart(chart_overview(hist, ticker), use_container_width=True, config={"displayModeBar": False})

    # ── Risk signals + key ratios side by side
    col_risk, col_ratio = st.columns([1, 1])

    with col_risk:
        st.markdown("#### ⚠️ 风险与信号")
        signals = compute_risk_signals(data)
        for s in signals:
            css = f"signal-{s['level']}"
            st.markdown(
                f'<div class="{css}">{s["icon"]} {s["label"]}</div>',
                unsafe_allow_html=True,
            )

    with col_ratio:
        st.markdown("#### 🏢 基本面速览")
        rows = {
            "毛利率":     fmt_pct(info.get("grossMargins")),
            "营业利润率": fmt_pct(info.get("operatingMargins")),
            "净利率":     fmt_pct(info.get("profitMargins")),
            "ROE":        fmt_pct(info.get("returnOnEquity")),
            "ROA":        fmt_pct(info.get("returnOnAssets")),
            "营收增长(YoY)": fmt_pct(info.get("revenueGrowth")),
            "净利增长(YoY)": fmt_pct(info.get("earningsGrowth")),
            "流动比率":   fmt_num(info.get("currentRatio")),
            "负债权益比": fmt_num(info.get("debtToEquity"), 1),
            "自由现金流": fmt_large(info.get("freeCashflow")),
        }
        df_ratio = pd.DataFrame(list(rows.items()), columns=["指标", "数值"])
        st.dataframe(df_ratio, hide_index=True, use_container_width=True,
                     height=min(40 * len(rows) + 50, 400))

    # ── Business summary
    summary = info.get("longBusinessSummary")
    if summary:
        with st.expander("公司简介"):
            st.write(summary)


# ════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL
# ════════════════════════════════════════════════════════════════
with tab2:
    period = st.radio(
        "周期", ["1个月", "3个月", "6个月", "1年"],
        index=2, horizontal=True,
    )

    st.plotly_chart(
        chart_technical(hist, ticker, period),
        use_container_width=True,
        config={"displayModeBar": True, "displaylogo": False},
    )

    # Technical summary table
    last = hist.iloc[-1]
    rsi_val = hist["RSI"].iloc[-1]
    macd_v  = hist["MACD"].iloc[-1]
    macd_s  = hist["MACD_Signal"].iloc[-1]
    macd_h  = hist["MACD_Hist"].iloc[-1]

    tech_data = {
        "当前价格": f"${cur_price:.2f}",
        "MA20":     f"${hist['MA20'].iloc[-1]:.2f}" if pd.notna(hist["MA20"].iloc[-1]) else "N/A",
        "MA50":     f"${hist['MA50'].iloc[-1]:.2f}" if pd.notna(hist["MA50"].iloc[-1]) else "N/A",
        "MA200":    f"${hist['MA200'].iloc[-1]:.2f}" if pd.notna(hist["MA200"].iloc[-1]) else "N/A",
        "价格 vs MA50":  ("↑ 高于" if cur_price > hist["MA50"].iloc[-1] else "↓ 低于") if pd.notna(hist["MA50"].iloc[-1]) else "N/A",
        "价格 vs MA200": ("↑ 高于" if cur_price > hist["MA200"].iloc[-1] else "↓ 低于") if pd.notna(hist["MA200"].iloc[-1]) else "N/A",
        "RSI (14)":  f"{rsi_val:.1f} — {'超买 ⚠️' if rsi_val>70 else '超卖 ℹ️' if rsi_val<30 else '中性 ✅'}" if pd.notna(rsi_val) else "N/A",
        "MACD":      f"{macd_v:.4f}",
        "MACD信号线": f"{macd_s:.4f}",
        "MACD柱":    f"{macd_h:.4f} ({'看涨 ✅' if macd_h > 0 else '看跌 ⚠️'})" if pd.notna(macd_h) else "N/A",
        "布林上轨": f"${hist['BB_Upper'].iloc[-1]:.2f}" if pd.notna(hist["BB_Upper"].iloc[-1]) else "N/A",
        "布林下轨": f"${hist['BB_Lower'].iloc[-1]:.2f}" if pd.notna(hist["BB_Lower"].iloc[-1]) else "N/A",
    }
    df_tech = pd.DataFrame(list(tech_data.items()), columns=["指标", "数值"])
    st.dataframe(df_tech, hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — FINANCIALS
# ════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        fig_inc = chart_income(fin)
        if fig_inc:
            st.plotly_chart(fig_inc, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("暂无收入利润数据")

    with col_b:
        fig_mg = chart_margins(fin)
        if fig_mg:
            st.plotly_chart(fig_mg, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("暂无利润率数据")

    col_c, col_d = st.columns(2)

    with col_c:
        fig_cf = chart_cashflow(fin)
        if fig_cf:
            st.plotly_chart(fig_cf, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("暂无现金流数据")

    with col_d:
        fig_bl = chart_balance(fin)
        if fig_bl:
            st.plotly_chart(fig_bl, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("暂无负债数据")

    # Raw numbers table
    all_years = sorted(
        {y for v in fin.values() if v for y in v.keys()},
        reverse=True,
    )[:4]

    if all_years:
        labels = {
            "revenue":          "总营收",
            "gross_profit":     "毛利润",
            "operating_income": "营业利润",
            "net_income":       "净利润",
            "ebitda":           "EBITDA",
            "operating_cf":     "经营现金流",
            "capex":            "资本支出",
            "free_cf":          "自由现金流",
            "total_assets":     "总资产",
            "total_debt":       "总负债",
            "total_equity":     "股东权益",
            "cash":             "现金",
        }
        rows = []
        for key, label in labels.items():
            d = fin.get(key) or {}
            row = {"项目（百万美元）": label}
            for y in all_years:
                row[y] = f"{d[y]:,.1f}" if y in d and d[y] is not None else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — NEWS
# ════════════════════════════════════════════════════════════════
with tab4:
    earnings = data.get("earnings_date")
    if earnings:
        st.info(f"📅 **下次财报日期：{earnings}**")

    analyst_rec = info.get("recommendationKey", "").upper()
    n_analysts  = info.get("numberOfAnalystOpinions")
    t_mean  = info.get("targetMeanPrice")
    t_high  = info.get("targetHighPrice")
    t_low   = info.get("targetLowPrice")

    if analyst_rec or t_mean:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("分析师评级", analyst_rec or "N/A")
        c2.metric("覆盖分析师数", str(n_analysts) if n_analysts else "N/A")
        c3.metric("目标价（均值）", f"${t_mean:.2f}" if t_mean else "N/A",
                  f"{(t_mean-cur_price)/cur_price*100:+.1f}% 空间" if t_mean else None)
        c4.metric("目标价区间", f"${t_low:.0f} – ${t_high:.0f}" if t_low and t_high else "N/A")

    st.markdown("")
    news = data.get("news", [])
    if news:
        st.markdown(f"#### 最新新闻（共 {len(news)} 条）")
        for item in news:
            st.markdown(
                f'<div class="news-card">'
                f'<strong>{item["title"]}</strong><br>'
                f'<small style="color:#6b7280">{item["publisher"]} · {item["date"]}</small>'
                + (f'<br><span style="color:#374151;font-size:0.9rem">{item["summary"]}</span>' if item.get("summary") else "")
                + f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("暂无最新新闻")
