"""
Paper-trading portfolio logic.

Data schema (stored in GitHub Gist via cloud.py):
{
  "holdings": [
    { "ticker", "market", "name", "shares", "avg_cost", "currency", "first_bought" }
  ],
  "transactions": [
    { "id", "ticker", "market", "name", "action"("buy"|"sell"),
      "shares", "price", "amount", "currency",
      "realized_pnl"(sell only), "date", "note" }
  ],
  "value_history": [
    { "date", "total_cost", "total_value", "pnl_pct" }
  ]
}
"""

import uuid
from datetime import datetime


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def get_holding(data: dict, ticker: str, market: str) -> dict | None:
    for h in data["holdings"]:
        if h["ticker"] == ticker and h["market"] == market:
            return h
    return None


def buy(data: dict, ticker: str, market: str, name: str,
        shares: float, price: float, currency: str, note: str = "") -> dict:
    """
    Execute a buy. Updates or creates the holding using weighted-average cost.
    Returns the mutated data dict.
    """
    h = get_holding(data, ticker, market)
    if h:
        total_cost  = h["shares"] * h["avg_cost"] + shares * price
        total_shares = h["shares"] + shares
        h["avg_cost"] = total_cost / total_shares
        h["shares"]   = round(total_shares, 8)
        h["name"]     = name or h.get("name", "")
    else:
        data["holdings"].append({
            "ticker":      ticker,
            "market":      market,
            "name":        name,
            "shares":      round(shares, 8),
            "avg_cost":    price,
            "currency":    currency,
            "first_bought": datetime.now().strftime("%Y-%m-%d"),
        })

    data["transactions"].append({
        "id":       str(uuid.uuid4())[:8],
        "ticker":   ticker,
        "market":   market,
        "name":     name,
        "action":   "buy",
        "shares":   shares,
        "price":    price,
        "amount":   round(shares * price, 4),
        "currency": currency,
        "date":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note":     note,
    })
    return data


def sell(data: dict, ticker: str, market: str,
         shares: float, price: float, note: str = "") -> tuple[dict, float | None, str]:
    """
    Execute a sell (AVCO method).
    Returns (updated_data, realized_pnl, error_message).
    realized_pnl is None and error_message is set on failure.
    """
    h = get_holding(data, ticker, market)
    if h is None:
        return data, None, f"持仓中没有 {ticker}"
    if shares > h["shares"] + 1e-8:
        return data, None, f"卖出数量 {shares} 超过持仓 {h['shares']:.4f}"

    realized = round((price - h["avg_cost"]) * shares, 4)

    data["transactions"].append({
        "id":           str(uuid.uuid4())[:8],
        "ticker":       ticker,
        "market":       market,
        "name":         h.get("name", ""),
        "action":       "sell",
        "shares":       shares,
        "price":        price,
        "amount":       round(shares * price, 4),
        "currency":     h.get("currency", "$"),
        "realized_pnl": realized,
        "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "note":         note,
    })

    h["shares"] = round(h["shares"] - shares, 8)
    if h["shares"] < 1e-6:
        data["holdings"] = [x for x in data["holdings"]
                            if not (x["ticker"] == ticker and x["market"] == market)]
    return data, realized, ""


# ── Value history snapshot ─────────────────────────────────────────────────────

def append_snapshot(data: dict, live_prices: dict) -> dict:
    """
    Append (or refresh today's) portfolio value snapshot.
    live_prices: {ticker: current_price_float}
    The snapshot is currency-agnostic — it sums raw numbers, which works for
    single-currency portfolios and gives a directionally correct P&L % for
    mixed-currency portfolios.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    snap  = _make_snapshot(today, data["holdings"], live_prices)

    history = data.get("value_history", [])
    if history and history[-1]["date"] == today:
        history[-1] = snap          # refresh today
    else:
        history.append(snap)

    data["value_history"] = history[-365:]   # keep 1 year
    return data


def _make_snapshot(date: str, holdings: list, prices: dict) -> dict:
    total_cost  = 0.0
    total_value = 0.0
    for h in holdings:
        p = prices.get(h["ticker"]) or h["avg_cost"]
        total_cost  += h["shares"] * h["avg_cost"]
        total_value += h["shares"] * p
    pnl_pct = round((total_value - total_cost) / total_cost * 100, 3) if total_cost else 0.0
    return {
        "date":        date,
        "total_cost":  round(total_cost, 2),
        "total_value": round(total_value, 2),
        "pnl_pct":     pnl_pct,
    }


# ── Aggregates ────────────────────────────────────────────────────────────────

def realized_pnl_total(data: dict) -> float:
    return sum(
        t.get("realized_pnl", 0) or 0
        for t in data.get("transactions", [])
        if t["action"] == "sell"
    )


def holdings_with_prices(holdings: list, prices: dict) -> list[dict]:
    """Enrich holding dicts with live price, market_value, pnl, pnl_pct."""
    rows = []
    for h in holdings:
        cur = prices.get(h["ticker"])
        cost_val = h["shares"] * h["avg_cost"]
        mkt_val  = h["shares"] * cur if cur is not None else None
        pnl      = round(mkt_val - cost_val, 2) if mkt_val is not None else None
        pnl_pct  = round(pnl / cost_val * 100, 2) if pnl is not None and cost_val else None
        rows.append({**h, "cur_price": cur, "market_value": mkt_val,
                     "pnl": pnl, "pnl_pct": pnl_pct})
    return rows
