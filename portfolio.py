import json
from pathlib import Path
from datetime import datetime

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"


def load() -> list[dict]:
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8")).get("holdings", [])
        except Exception:
            return []
    return []


def save(holdings: list[dict]):
    PORTFOLIO_FILE.write_text(
        json.dumps({"holdings": holdings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert(ticker: str, market: str, exchange: str, shares: float,
           cost_price: float, name: str = "", currency: str = "$") -> str:
    """Add or update a holding. Returns 'added' or 'updated'."""
    holdings = load()
    for h in holdings:
        if h["ticker"] == ticker and h["market"] == market:
            h.update({
                "shares": shares,
                "cost_price": cost_price,
                "name": name or h.get("name", ""),
                "currency": currency,
                "exchange": exchange,
            })
            save(holdings)
            return "updated"
    holdings.append({
        "ticker":     ticker,
        "market":     market,
        "exchange":   exchange,
        "name":       name,
        "shares":     shares,
        "cost_price": cost_price,
        "currency":   currency,
        "added":      datetime.now().strftime("%Y-%m-%d"),
    })
    save(holdings)
    return "added"


def remove(ticker: str, market: str):
    save([h for h in load() if not (h["ticker"] == ticker and h["market"] == market)])
