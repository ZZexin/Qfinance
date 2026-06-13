import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data import fetch_stock_data, compute_risk_signals, fetch_current_prices, compute_composite_score
import cloud
import portfolio as pf

st.set_page_config(
    page_title="秋秋炒股工具",
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


# ── signal chart builders ─────────────────────────────────────────────────────

def chart_gauge(score: float, direction: str, credibility: float, color: str) -> go.Figure:
    """Half-gauge (0-100) composite score indicator."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 40, "color": color}, "suffix": ""},
        title={"text": f"<b style='font-size:1.1em'>{direction}</b>"
                       f"<br><span style='font-size:0.75em;color:#6b7280'>"
                       f"综合可信度 {credibility:.0f}%</span>"},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["强空", "看空", "中性", "看多", "强多"],
                "tickfont": {"size": 10},
            },
            "bar":    {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  32],  "color": "rgba(220,38,38,0.12)"},
                {"range": [32, 43],  "color": "rgba(239,68,68,0.07)"},
                {"range": [43, 57],  "color": "rgba(156,163,175,0.07)"},
                {"range": [57, 68],  "color": "rgba(34,197,94,0.07)"},
                {"range": [68, 100], "color": "rgba(22,163,74,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def chart_rec_breakdown(counts: dict, n_analysts: int) -> go.Figure:
    """Stacked horizontal bar: strongBuy / buy / hold / sell / strongSell."""
    cats   = ["强卖", "卖出", "持有", "买入", "强买"]
    clrs   = ["#dc2626", "#f87171", "#94a3b8", "#4ade80", "#16a34a"]
    vals   = [counts.get(c, 0) for c in cats]
    total  = sum(vals) or 1

    fig = go.Figure()
    for cat, val, clr in zip(cats, vals, clrs):
        pct = val / total * 100
        fig.add_trace(go.Bar(
            name=cat, x=[val], y=["评级"],
            orientation="h",
            marker_color=clr,
            text=f"{cat} {val}" if val > 0 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"{cat}: {val}人 ({pct:.0f}%)<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        height=90,
        margin=dict(l=0, r=0, t=10, b=35),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.6,
                    font=dict(size=11)),
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, total * 1.02]),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"共 {n_analysts} 位分析师",
            x=0.5, y=1.3, xref="paper", yref="paper",
            showarrow=False, font=dict(size=11, color="#6b7280"),
        )],
    )
    return fig


def chart_target_range(cur: float, t_low: float, t_mean: float,
                        t_high: float, currency: str = "$") -> go.Figure:
    """Bullet chart: current price vs analyst target price range."""
    fig = go.Figure()

    # Range bar (low → high)
    fig.add_trace(go.Scatter(
        x=[t_low, t_high], y=[0, 0],
        mode="lines",
        line=dict(color="#94a3b8", width=14),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Mean target
    fig.add_trace(go.Scatter(
        x=[t_mean], y=[0],
        mode="markers+text",
        marker=dict(color="#3b82f6", size=16, symbol="diamond"),
        text=[f"目标均价<br>{currency}{t_mean:.2f}"],
        textposition="top center",
        name="分析师目标价（均）",
        hovertemplate=f"目标均价: {currency}{t_mean:.2f}<extra></extra>",
    ))
    # Current price
    fig.add_trace(go.Scatter(
        x=[cur], y=[0],
        mode="markers+text",
        marker=dict(color="#ef5350", size=16, symbol="circle"),
        text=[f"现价<br>{currency}{cur:.2f}"],
        textposition="bottom center",
        name="当前价格",
        hovertemplate=f"当前价: {currency}{cur:.2f}<extra></extra>",
    ))
    # Low / High labels
    for val, label in [(t_low, f"低 {currency}{t_low:.0f}"), (t_high, f"高 {currency}{t_high:.0f}")]:
        fig.add_annotation(x=val, y=0, text=label, showarrow=False,
                           yshift=-28, font=dict(size=10, color="#6b7280"))

    fig.update_layout(
        height=160,
        margin=dict(l=20, r=20, t=10, b=40),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.5,
                    font=dict(size=11)),
        xaxis=dict(showgrid=False, showticklabels=False,
                   range=[t_low * 0.92, t_high * 1.08]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-0.5, 0.6]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def chart_signal_components(components: dict) -> go.Figure:
    """Horizontal bar chart showing each sub-signal score with credibility annotation."""
    labels = [c["label"] for c in components.values()]
    scores = [c["score"] for c in components.values()]
    creds  = [c["cred"]  for c in components.values()]
    weights= [c["weight"] for c in components.values()]

    bar_colors = [
        "#16a34a" if s >= 68 else "#22c55e" if s >= 57 else
        "#94a3b8" if s >= 43 else "#f87171" if s >= 32 else "#dc2626"
        for s in scores
    ]
    texts = [
        f"  {s:.0f}/100  （可信度 {c:.0f}%，权重 {w*100:.0f}%）"
        for s, c, w in zip(scores, creds, weights)
    ]

    fig = go.Figure(go.Bar(
        y=labels, x=scores, orientation="h",
        marker_color=bar_colors,
        text=texts,
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
    ))
    fig.add_vline(x=50, line=dict(color="#9ca3af", dash="dash", width=1))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=160, t=10, b=10),
        xaxis=dict(range=[0, 120], showgrid=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


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

st.markdown("## 📊 秋秋炒股工具 · 股票数据仪表板")
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

# ── Tabs — always rendered so 💼 tab is reachable without a stock search ──────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 概览", "🕯 K线 & 技术指标", "📑 财务报表", "📰 新闻 & 事件", "🔍 信号分析", "💼 模拟持仓"]
)

if data is not None:
    info       = data["info"]
    hist       = data["hist"]
    fin        = data["financials"]
    ticker     = data["ticker"]
    cur_price  = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur_price
    day_chg    = (cur_price - prev_close) / prev_close * 100
    company_name = info.get("longName") or info.get("shortName") or ticker
    st.markdown(f"### {company_name}（{ticker}）")
    st.caption(f"{info.get('sector','—')} · {info.get('industry','—')} · {info.get('country','—')}")


# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:
  if data is None:
    st.info("请在上方输入股票代码并查询")
  if data is not None:

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
  if data is None:
    st.info("请在上方输入股票代码并查询")
  if data is not None:
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
  if data is None:
    st.info("请在上方输入股票代码并查询")
  if data is not None:
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
  if data is None:
    st.info("请在上方输入股票代码并查询")
  if data is not None:
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


# ════════════════════════════════════════════════════════════════
# TAB 5 — MULTI-SOURCE SIGNAL ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab5:
  if data is None:
    st.info("请在上方输入股票代码并查询")
  if data is not None:
    signals_raw = data.get("signals", {})
    score_data  = compute_composite_score(info, signals_raw, hist)

    comp       = score_data["components"]
    currency   = data.get("currency", "$")

    # ── Row 1: Composite gauge + component bar ─────────────────────
    col_gauge, col_bar = st.columns([1, 1.4])

    with col_gauge:
        st.markdown("#### 综合信号评分")
        st.plotly_chart(
            chart_gauge(
                score_data["composite"],
                score_data["direction"],
                score_data["credibility"],
                score_data["color"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        # Mini legend boxes
        legend_items = [
            ("强烈看多 ≥68", "#16a34a"), ("看多 57–68", "#22c55e"),
            ("中性 43–57", "#94a3b8"),   ("看空 32–43", "#f87171"),
            ("强烈看空 <32", "#dc2626"),
        ]
        legend_html = "".join(
            f"<span style='display:inline-block;width:10px;height:10px;"
            f"background:{c};border-radius:2px;margin-right:4px;margin-left:8px'></span>"
            f"<span style='font-size:0.75rem;color:#6b7280'>{lbl}</span>"
            for lbl, c in legend_items
        )
        st.markdown(legend_html, unsafe_allow_html=True)

    with col_bar:
        st.markdown("#### 各维度评分明细")
        st.caption("每条：0=极度看空  50=中性  100=极度看多  |  括号内为当前信号对综合分的可信度与权重")
        st.plotly_chart(
            chart_signal_components(comp),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.divider()

    # ── Row 2: Analyst consensus + target price ────────────────────
    col_rec, col_tp = st.columns([1, 1])

    with col_rec:
        st.markdown("#### 📊 分析师评级共识")
        c_analyst  = comp["analyst"]
        counts     = c_analyst["counts"]
        n_ana      = c_analyst["n_analysts"]
        if sum(counts.values()) > 0:
            st.plotly_chart(
                chart_rec_breakdown(counts, n_ana),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            # Recommendation key from info
            rec_key = info.get("recommendationKey", "")
            rec_mean = info.get("recommendationMean")
            if rec_key:
                key_map = {
                    "strong_buy": ("强烈买入", "#16a34a"),
                    "buy":        ("买入",     "#22c55e"),
                    "hold":       ("持有",     "#f59e0b"),
                    "sell":       ("卖出",     "#ef4444"),
                    "strong_sell":("强烈卖出", "#dc2626"),
                }
                lbl, clr = key_map.get(rec_key.lower(), (rec_key, "#6b7280"))
                mean_txt = f"  （评级均值 {rec_mean:.2f}/5.0）" if rec_mean else ""
                st.markdown(
                    f"<div style='background:{clr}22;border:1px solid {clr}66;"
                    f"border-radius:6px;padding:8px 14px;font-size:0.9rem'>"
                    f"<b style='color:{clr}'>综合建议：{lbl}</b>{mean_txt}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("暂无分析师评级数据（该股票可能覆盖较少或为非美股）")

    with col_tp:
        st.markdown("#### 🎯 分析师目标价区间")
        target   = info.get("targetMeanPrice")
        t_low    = info.get("targetLowPrice")
        t_high   = info.get("targetHighPrice")
        upside   = score_data.get("upside_pct")
        if target and t_low and t_high:
            st.plotly_chart(
                chart_target_range(cur_price, t_low, target, t_high, currency),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            up_clr = "#22c55e" if (upside or 0) >= 0 else "#ef4444"
            st.markdown(
                f"<div style='font-size:0.88rem;color:#6b7280'>"
                f"目标价区间：{currency}{t_low:.2f} – {currency}{t_high:.2f}  |  "
                f"均值目标价：{currency}{target:.2f}  |  "
                f"<b style='color:{up_clr}'>潜在空间：{upside:+.1f}%</b></div>" if upside is not None else
                f"目标价均值：{currency}{target:.2f}",
                unsafe_allow_html=True,
            )
        else:
            st.info("暂无目标价数据")

    st.divider()

    # ── Row 3: Rating changes + insider activity ───────────────────
    col_ud, col_ins = st.columns([1, 1])

    with col_ud:
        st.markdown("#### 📋 近6个月评级变动")
        c_ratings = comp["ratings"]
        ud_list   = signals_raw.get("upgrades_downgrades") or []
        up_n, dn_n, init_n = c_ratings["upgrades"], c_ratings["downgrades"], c_ratings["initiations"]

        if ud_list:
            # Summary chips
            chips_html = ""
            if up_n:
                chips_html += (
                    f"<span style='background:#d1fae5;border:1px solid #6ee7b7;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem;"
                    f"margin-right:6px'>⬆️ 上调 {up_n}</span>"
                )
            if dn_n:
                chips_html += (
                    f"<span style='background:#fee2e2;border:1px solid #fca5a5;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem;"
                    f"margin-right:6px'>⬇️ 下调 {dn_n}</span>"
                )
            if init_n:
                chips_html += (
                    f"<span style='background:#dbeafe;border:1px solid #93c5fd;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem;"
                    f"margin-right:6px'>🆕 首评 {init_n}</span>"
                )
            if chips_html:
                st.markdown(chips_html, unsafe_allow_html=True)
                st.markdown("")

            # Table
            date_col  = list(ud_list[0].keys())[0]   # first col is the date
            rows = []
            for r in ud_list[:12]:
                action_raw = str(r.get("Action", ""))
                action_map = {
                    "up": "⬆️ 上调", "down": "⬇️ 下调",
                    "init": "🆕 首评", "reit": "🔄 重申",
                    "main": "✅ 维持", "upgrade": "⬆️ 上调", "downgrade": "⬇️ 下调",
                }
                action_lbl = action_map.get(action_raw.lower(), action_raw)
                rows.append({
                    "日期":   str(r.get(date_col, ""))[:10],
                    "机构":   str(r.get("Firm", r.get("firm", ""))),
                    "动作":   action_lbl,
                    "新评级": str(r.get("ToGrade",   r.get("toGrade",   ""))),
                    "原评级": str(r.get("FromGrade", r.get("fromGrade", ""))),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=300)
        else:
            st.info("暂无近期评级变动记录")

    with col_ins:
        st.markdown("#### 🏠 内部人交易（近期）")
        c_insider  = comp["insider"]
        txn_list   = signals_raw.get("insider_transactions") or []
        net_val    = c_insider["net_value"]
        buy_n, sell_n = c_insider["buy_count"], c_insider["sell_count"]

        if txn_list:
            # Net summary chip
            if net_val > 1e5:
                chip_html = (
                    f"<span style='background:#d1fae5;border:1px solid #6ee7b7;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem'>"
                    f"💚 净买入 +{currency}{abs(net_val)/1e6:.2f}M（买入{buy_n}笔 / 卖出{sell_n}笔）</span>"
                )
            elif net_val < -1e5:
                chip_html = (
                    f"<span style='background:#fee2e2;border:1px solid #fca5a5;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem'>"
                    f"🔴 净卖出 -{currency}{abs(net_val)/1e6:.2f}M（买入{buy_n}笔 / 卖出{sell_n}笔）</span>"
                )
            else:
                chip_html = (
                    f"<span style='background:#f3f4f6;border:1px solid #d1d5db;"
                    f"border-radius:12px;padding:3px 10px;font-size:0.8rem'>"
                    f"⚪ 净活动较小（买入{buy_n}笔 / 卖出{sell_n}笔）</span>"
                )
            st.markdown(chip_html, unsafe_allow_html=True)
            st.markdown("")

            rows = []
            for t in txn_list[:10]:
                tx_str = str(t.get("Transaction", "") or "")
                if any(w in tx_str.lower() for w in ["purchase", "buy", "acqui"]):
                    icon = "💚"
                elif any(w in tx_str.lower() for w in ["sale", "sell", "dispos"]):
                    icon = "🔴"
                else:
                    icon = "⚪"
                rows.append({
                    "日期":   str(t.get("Date",     t.get("date",     "")))[:10],
                    "内部人": str(t.get("Insider",  t.get("insider",  ""))),
                    "职位":   str(t.get("Position", t.get("position", ""))),
                    "操作":   f"{icon} {tx_str}",
                    "价值($)":str(t.get("Value",    t.get("value",    ""))),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=300)
        else:
            st.info("暂无内部人交易记录（通常仅美股 SEC 披露）")

    st.divider()

    # ── Row 4: Institutional holders ──────────────────────────────
    st.markdown("#### 🏦 主要机构持仓")
    inst_col, major_col = st.columns([2, 1])

    with inst_col:
        inst_list = signals_raw.get("institutional_holders") or []
        if inst_list:
            df_inst = pd.DataFrame(inst_list)
            # Rename common column patterns
            rename_map = {
                "Holder": "机构名称", "holder": "机构名称",
                "Shares": "持股数量", "shares": "持股数量",
                "Date Reported": "报告日期", "dateReported": "报告日期",
                "% Out": "持股比例", "pctHeld": "持股比例",
                "Value": "持仓价值", "value": "持仓价值",
            }
            df_inst = df_inst.rename(columns={k: v for k, v in rename_map.items() if k in df_inst.columns})
            st.dataframe(df_inst, hide_index=True, use_container_width=True, height=300)
        else:
            st.info("暂无机构持仓数据")

    with major_col:
        major_list = signals_raw.get("major_holders") or []
        if major_list:
            st.markdown("**持股结构**")
            for row in major_list:
                vals = list(row.values())
                if len(vals) >= 2:
                    pct_val, pct_lbl = str(vals[0]), str(vals[1])
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:5px 0;border-bottom:1px solid #f3f4f6;font-size:0.85rem'>"
                        f"<span style='color:#6b7280'>{pct_lbl}</span>"
                        f"<b>{pct_val}</b></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("暂无持股结构数据")

    # ── Credibility footnote ───────────────────────────────────────
    st.markdown("")
    st.caption(
        "💡 可信度说明：综合可信度由各信号的分析师覆盖数、评级变动频率、"
        "内部人交易数量及目标价收窄程度共同决定，数据越多覆盖越广可信度越高。"
        "本工具仅供参考，不构成投资建议。"
    )


# ════════════════════════════════════════════════════════════════
# TAB 6 — PAPER TRADING PORTFOLIO
# ════════════════════════════════════════════════════════════════
with tab6:

    # ── Session state init ────────────────────────────────────────
    for k, v in [("pf_token", ""), ("pf_gist_id", ""),
                 ("pf_data", None), ("pf_connected", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-load saved credentials on first visit
    if not st.session_state.pf_connected:
        tok, gid = cloud.load_credentials()
        if tok and gid:
            st.session_state.pf_token   = tok
            st.session_state.pf_gist_id = gid
            try:
                st.session_state.pf_data      = cloud.load(tok, gid)
                st.session_state.pf_connected = True
            except Exception:
                pass

    # ── Already connected: show minimal status bar, no expander needed ──
    if st.session_state.pf_connected:
        cb1, cb2 = st.columns([5, 1])
        cb1.caption(f"☁️ 已连接 Gist `{st.session_state.pf_gist_id}`")
        if cb2.button("断开", use_container_width=True, key="disconnect"):
            st.session_state.update({"pf_connected": False, "pf_token": "",
                                      "pf_gist_id": "", "pf_data": None})
            st.rerun()

    # ── Not connected: show one-time setup wizard ─────────────────
    else:
        st.markdown("""
### ☁️ 一次配置，永久免登录

持仓数据存在你的 **GitHub 私有 Gist**，只需设置一次。

#### 第一步：生成 GitHub Token
> 打开 **[github.com/settings/tokens](https://github.com/settings/tokens/new)**
> → Note 填 `qfinance`
> → 勾选 **`gist`** 权限（Classic token）
> → Generate token，复制 `ghp_...` 开头的内容
""")

        inp_token = st.text_input("粘贴 GitHub Token",
                                   value=st.session_state.pf_token,
                                   type="password", placeholder="ghp_xxxxxxxxxxxx",
                                   key="inp_token")
        inp_gist  = st.text_input("Gist ID（首次留空，自动创建）",
                                   value=st.session_state.pf_gist_id,
                                   placeholder="留空则自动创建",
                                   key="inp_gist")

        if inp_token.startswith("github_pat_"):
            st.warning(
                "检测到 Fine-grained token，需要在 **Account permissions → Gists → Read and write** 勾选权限。\n"
                "建议改用 Classic token（`ghp_` 开头）更简单。"
            )

        if st.button("🔗 连接 / 创建 Gist", use_container_width=True,
                     type="primary", key="connect_btn"):
            if not inp_token:
                st.warning("请粘贴 GitHub Token")
            else:
                with st.spinner("验证 Token..."):
                    ok, user_or_err = cloud.test_token(inp_token)
                if not ok:
                    st.error(f"❌ Token 无效：{user_or_err}")
                else:
                    gist_id = inp_gist.strip()
                    try:
                        if not gist_id:
                            with st.spinner("创建私有 Gist..."):
                                gist_id = cloud.create_gist(inp_token)
                        with st.spinner("加载数据..."):
                            pf_data = cloud.load(inp_token, gist_id)
                    except PermissionError as e:
                        st.error(f"❌ {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ 连接失败：{e}")
                        st.stop()

                    cloud.save_credentials(inp_token, gist_id)   # local; may no-op on Cloud
                    st.session_state.update({
                        "pf_token": inp_token, "pf_gist_id": gist_id,
                        "pf_data": pf_data, "pf_connected": True,
                    })

                    # ── Step 2 instructions: paste into Streamlit Cloud Secrets ──
                    st.success(f"✅ 连接成功！GitHub @{user_or_err}  ·  Gist: `{gist_id}`")
                    st.markdown(f"""
#### 第二步：保存到 Streamlit Cloud（**只需做一次**）

复制以下两行，粘贴到：
**Streamlit Cloud → 你的 App → ⋮ → Settings → Secrets**

```toml
github_token = "{inp_token}"
gist_id = "{gist_id}"
```

保存后刷新页面，以后打开 App 会**自动登录**，不需要再输入任何内容。
""")
                    st.button("✅ 我已保存，继续", use_container_width=True,
                              key="after_save", on_click=st.rerun)
                    st.stop()

        st.info("配置完成后，你的持仓数据将安全地存储在 GitHub 私有 Gist，任何设备打开 App 都能自动读取。")
        st.stop()

    # ── Shorthand ─────────────────────────────────────────────────
    pf_data  = st.session_state.pf_data
    pf_tok   = st.session_state.pf_token
    pf_gid   = st.session_state.pf_gist_id
    holdings = pf_data.get("holdings", [])

    def _save():
        cloud.save(pf_tok, pf_gid, pf_data)
        st.session_state.pf_data = pf_data

    # ── Refresh live prices & snapshot ────────────────────────────
    if holdings:
        with st.spinner("刷新实时价格..."):
            live_prices = fetch_current_prices(holdings)
        pf.append_snapshot(pf_data, live_prices)
        _save()
    else:
        live_prices = {}

    enriched = pf.holdings_with_prices(holdings, live_prices)

    # ── Summary metrics ───────────────────────────────────────────
    total_cost  = sum(h["shares"] * h["avg_cost"] for h in holdings)
    total_value = sum(h.get("market_value") or h["shares"] * h["avg_cost"]
                      for h in enriched)
    total_pnl   = total_value - total_cost
    total_pnl_p = total_pnl / total_cost * 100 if total_cost else 0
    realized    = pf.realized_pnl_total(pf_data)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("持仓数量",   f"{len(holdings)} 只")
    mc2.metric("总投入成本", f"{total_cost:,.0f}")
    mc3.metric("当前市值",   f"{total_value:,.0f}",
               f"{total_pnl:+,.0f}")
    mc4.metric("浮动盈亏%",  f"{total_pnl_p:+.2f}%",
               delta_color="normal" if total_pnl_p >= 0 else "inverse")
    mc5.metric("累计已实现盈亏", f"{realized:+,.2f}")

    # ── P&L history chart ─────────────────────────────────────────
    history = pf_data.get("value_history", [])
    if len(history) >= 2:
        hdf      = pd.DataFrame(history)
        is_up    = float(hdf["pnl_pct"].iloc[-1]) >= 0
        clr      = "#26a69a" if is_up else "#ef5350"
        fig_pnl  = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=hdf["date"], y=hdf["pnl_pct"],
            mode="lines", name="浮动盈亏%",
            line=dict(color=clr, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({'38,166,154' if is_up else '239,83,80'},0.10)",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ))
        fig_pnl.add_hline(y=0, line=dict(color="#94a3b8", width=1, dash="dot"))
        fig_pnl.update_layout(
            **PLOT_LAYOUT, height=260,
            title=dict(text="持仓总盈亏 % 历史", x=0.01),
            yaxis=dict(ticksuffix="%", gridcolor="#f3f4f6"),
            xaxis=dict(gridcolor="#f3f4f6"),
            showlegend=False,
        )
        st.plotly_chart(fig_pnl, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Holdings table ────────────────────────────────────────────
    st.markdown("#### 📋 当前持仓")
    if enriched:
        rows = []
        for h in enriched:
            ccy = h.get("currency", "$")
            pnl_p = h.get("pnl_pct")
            rows.append({
                "代码":    h["ticker"],
                "市场":    h.get("market", "—"),
                "名称":    (h.get("name") or "")[:12],
                "股数":    h["shares"],
                "均价":    f"{ccy}{h['avg_cost']:.2f}",
                "现价":    f"{ccy}{h['cur_price']:.2f}" if h["cur_price"] else "—",
                "市值":    f"{ccy}{h['market_value']:,.0f}" if h["market_value"] else "—",
                "浮盈":    f"{ccy}{h['pnl']:+,.2f}" if h["pnl"] is not None else "—",
                "浮盈%":   f"{pnl_p:+.2f}%" if pnl_p is not None else "—",
                "买入日":  h.get("first_bought", "—"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True,
                     height=min(42 * len(rows) + 55, 380))
    else:
        st.info("暂无持仓，在下方买入第一只股票吧")

    st.divider()

    # ── Trade form ────────────────────────────────────────────────
    st.markdown("#### 📝 模拟交易")
    buy_tab, sell_tab = st.tabs(["🟢 买入", "🔴 卖出"])

    with buy_tab:
        bc1, bc2 = st.columns(2)
        b_mkt    = bc1.selectbox("市场", ["US", "AU", "CN（沪）", "CN（深）", "HK"],
                                  key="b_mkt")
        b_raw    = bc2.text_input("股票代码",
                                   value=data["ticker"] if data else "",
                                   placeholder="如 AAPL / 600519",
                                   key="b_raw")

        # Normalize ticker
        _sfx = {"AU": ".AX", "CN（沪）": ".SS", "CN（深）": ".SZ", "HK": ".HK", "US": ""}
        b_ticker = (b_raw.strip().upper() + _sfx.get(b_mkt, "")) if b_raw else ""
        b_mkt_key = b_mkt.split("（")[0]   # "US" / "AU" / "CN" / "HK"

        bc3, bc4, bc5 = st.columns(3)
        _ccy_map = {"US": "$", "AU": "A$", "CN（沪）": "¥", "CN（深）": "¥", "HK": "HK$"}
        b_ccy    = _ccy_map.get(b_mkt, "$")

        # Default price = live price of current queried stock, else 0
        _default_price = 0.0
        if data and b_ticker == data.get("ticker"):
            _default_price = float(data["hist"]["Close"].iloc[-1])

        b_shares = bc3.number_input("股数", min_value=0.0001, value=1.0,
                                     format="%.4f", key="b_shares")
        b_price  = bc4.number_input(f"成交价（{b_ccy}）",
                                     min_value=0.0001, value=max(_default_price, 0.01),
                                     format="%.4f", key="b_price")
        b_note   = bc5.text_input("备注", placeholder="可选", key="b_note")

        b_total = b_shares * b_price
        st.caption(f"交易金额：{b_ccy}{b_total:,.2f}")

        if st.button("✅ 确认买入", use_container_width=True, type="primary", key="do_buy"):
            if not b_ticker:
                st.warning("请输入股票代码")
            else:
                # Try to get company name
                b_name = ""
                try:
                    import yfinance as _yf
                    _info = _yf.Ticker(b_ticker).info
                    b_name = _info.get("shortName") or _info.get("longName") or ""
                except Exception:
                    pass
                pf.buy(pf_data, b_ticker, b_mkt_key, b_name,
                       b_shares, b_price, b_ccy, b_note)
                _save()
                st.success(f"✅ 买入 {b_ticker} × {b_shares} @ {b_ccy}{b_price:.4f}")
                st.rerun()

    with sell_tab:
        if not holdings:
            st.info("暂无持仓可卖出")
        else:
            ticker_opts = [f"{h['ticker']} ({h['market']})" for h in holdings]
            sel = st.selectbox("选择持仓", ticker_opts, key="s_sel")
            idx = ticker_opts.index(sel)
            sh  = holdings[idx]
            ccy = sh.get("currency", "$")
            cur = live_prices.get(sh["ticker"])

            sc1, sc2, sc3 = st.columns(3)
            s_shares = sc1.number_input("卖出股数",
                                         min_value=0.0001, max_value=float(sh["shares"]),
                                         value=float(sh["shares"]), format="%.4f", key="s_shares")
            s_price  = sc2.number_input(f"成交价（{ccy}）",
                                         min_value=0.0001,
                                         value=float(cur) if cur else sh["avg_cost"],
                                         format="%.4f", key="s_price")
            s_note   = sc3.text_input("备注", placeholder="可选", key="s_note")

            est_pnl = (s_price - sh["avg_cost"]) * s_shares
            pnl_color = "🟢" if est_pnl >= 0 else "🔴"
            st.caption(
                f"持仓均价 {ccy}{sh['avg_cost']:.4f}  ·  "
                f"预计{'盈利' if est_pnl >= 0 else '亏损'} {pnl_color} "
                f"{ccy}{est_pnl:+,.2f}"
            )

            if st.button("✅ 确认卖出", use_container_width=True, type="primary", key="do_sell"):
                _, realized, err = pf.sell(pf_data, sh["ticker"], sh["market"],
                                            s_shares, s_price, s_note)
                if err:
                    st.error(err)
                else:
                    _save()
                    st.success(
                        f"✅ 卖出 {sh['ticker']} × {s_shares}  "
                        f"已实现盈亏 {ccy}{realized:+,.2f}"
                    )
                    st.rerun()

    # ── Transaction history ───────────────────────────────────────
    txns = pf_data.get("transactions", [])
    with st.expander(f"📜 交易记录（共 {len(txns)} 笔）", expanded=False):
        if txns:
            tx_rows = []
            for t in reversed(txns):
                ccy = t.get("currency", "$")
                pnl = t.get("realized_pnl")
                tx_rows.append({
                    "日期":   t["date"],
                    "操作":   "🟢 买入" if t["action"] == "buy" else "🔴 卖出",
                    "代码":   t["ticker"],
                    "名称":   (t.get("name") or "")[:10],
                    "股数":   t["shares"],
                    "价格":   f"{ccy}{t['price']:.4f}",
                    "金额":   f"{ccy}{t['amount']:,.2f}",
                    "已实现盈亏": f"{ccy}{pnl:+,.2f}" if pnl is not None else "—",
                    "备注":   t.get("note", ""),
                })
            st.dataframe(pd.DataFrame(tx_rows), hide_index=True,
                         use_container_width=True)
        else:
            st.info("暂无交易记录")
