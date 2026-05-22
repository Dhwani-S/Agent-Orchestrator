"""
MCP server.

Nine tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file

web_search:  Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results.
fetch_url:   crawl4ai only — clean markdown via headless Chromium.
Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

load_dotenv(Path(__file__).parent / ".env")

mcp = FastMCP("eagv3-s6-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _ddg_search(query: str, max_results: int) -> list[dict]:
    hits: list[dict] = []
    with DDGS() as ddgs:
        for backend in ("auto", "html", "lite"):
            try:
                hits = list(ddgs.text(query, max_results=max_results, backend=backend))
            except Exception:
                hits = []
            if hits:
                break
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]


async def _crawl4ai_fetch(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url)
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
    # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
    # serializes as {} because its real field is private. Pull the raw string
    # out and force a plain str so FastMCP serializes correctly.
    md = r.markdown
    raw = (
        getattr(md, "raw_markdown", None)
        or getattr(md, "fit_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (Tavily primary, DDG fallback). Hard-capped at 5 results. Example: web_search("python asyncio tutorial", 3)."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")
    results = _ddg_search(query, max_results)
    _bump("duckduckgo")
    return results


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> dict:
    """Fetch clean markdown from a URL via crawl4ai (headless Chromium). Example: fetch_url("https://example.com")."""
    return await _crawl4ai_fetch(url)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Example: get_time("Asia/Kolkata")."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": timezone,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


# ── Mock Travel Booking Tools ──────────────────────────────────────

BOOKINGS_FILE = Path(__file__).parent / "state" / "bookings.json"

def _load_bookings() -> list[dict]:
    BOOKINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if BOOKINGS_FILE.exists():
        return json.loads(BOOKINGS_FILE.read_text(encoding="utf-8"))
    return []

def _save_bookings(bookings: list[dict]):
    BOOKINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BOOKINGS_FILE.write_text(json.dumps(bookings, indent=2, default=str), encoding="utf-8")


MOCK_HOTELS = {
    "mumbai": [
        {"name": "Trident BKC", "area": "Bandra Kurla Complex", "price_per_night": 3500, "rating": 4.3, "distance_to_jio_world_centre_km": 1.2, "amenities": ["wifi", "pool", "gym", "restaurant", "airport shuttle"]},
        {"name": "ITC Maratha", "area": "Andheri East", "price_per_night": 4000, "rating": 4.5, "distance_to_jio_world_centre_km": 8.5, "amenities": ["wifi", "pool", "spa", "restaurant", "business centre"]},
        {"name": "Hyatt Regency", "area": "Andheri East", "price_per_night": 3800, "rating": 4.2, "distance_to_jio_world_centre_km": 7.0, "amenities": ["wifi", "pool", "gym", "restaurant"]},
        {"name": "Sofitel BKC", "area": "Bandra Kurla Complex", "price_per_night": 4200, "rating": 4.6, "distance_to_jio_world_centre_km": 0.8, "amenities": ["wifi", "pool", "spa", "gym", "restaurant", "lounge"]},
        {"name": "Courtyard by Marriott", "area": "Andheri East", "price_per_night": 2800, "rating": 4.0, "distance_to_jio_world_centre_km": 9.0, "amenities": ["wifi", "gym", "restaurant"]},
        {"name": "The Leela Mumbai", "area": "Andheri East", "price_per_night": 3200, "rating": 4.4, "distance_to_jio_world_centre_km": 8.0, "amenities": ["wifi", "pool", "spa", "restaurant", "business centre"]},
        {"name": "JW Marriott Sahar", "area": "Andheri East", "price_per_night": 3600, "rating": 4.3, "distance_to_jio_world_centre_km": 10.0, "amenities": ["wifi", "pool", "gym", "spa", "restaurant"]},
        {"name": "Holiday Inn BKC", "area": "Bandra Kurla Complex", "price_per_night": 2500, "rating": 3.8, "distance_to_jio_world_centre_km": 1.5, "amenities": ["wifi", "gym", "restaurant"]},
    ],
    "jamnagar": [
        {"name": "Hotel President", "area": "City Centre", "price_per_night": 1800, "rating": 3.5, "amenities": ["wifi", "restaurant"]},
        {"name": "The Fern Residency", "area": "Patel Colony", "price_per_night": 2500, "rating": 3.9, "amenities": ["wifi", "gym", "restaurant"]},
        {"name": "Aram Resort", "area": "Highway", "price_per_night": 3000, "rating": 4.0, "amenities": ["wifi", "pool", "restaurant", "garden"]},
    ],
    "bangalore": [
        {"name": "Taj MG Road", "area": "MG Road", "price_per_night": 3500, "rating": 4.4, "amenities": ["wifi", "pool", "spa", "restaurant"]},
        {"name": "ITC Gardenia", "area": "Residency Road", "price_per_night": 4000, "rating": 4.5, "amenities": ["wifi", "pool", "spa", "gym", "restaurant"]},
    ],
}

MOCK_FLIGHTS = [
    {"flight": "AI-801", "airline": "Air India", "from": "Bangalore", "to": "Mumbai", "depart": "06:15", "arrive": "08:20", "price": 4500, "class": "economy"},
    {"flight": "6E-302", "airline": "IndiGo", "from": "Bangalore", "to": "Mumbai", "depart": "07:45", "arrive": "09:50", "price": 3200, "class": "economy"},
    {"flight": "UK-852", "airline": "Vistara", "from": "Bangalore", "to": "Mumbai", "depart": "09:30", "arrive": "11:35", "price": 5100, "class": "economy"},
    {"flight": "SG-171", "airline": "SpiceJet", "from": "Bangalore", "to": "Mumbai", "depart": "14:00", "arrive": "16:05", "price": 2800, "class": "economy"},
    {"flight": "AI-671", "airline": "Air India", "from": "Mumbai", "to": "Jamnagar", "depart": "10:30", "arrive": "12:00", "price": 3800, "class": "economy"},
    {"flight": "6E-517", "airline": "IndiGo", "from": "Mumbai", "to": "Jamnagar", "depart": "15:20", "arrive": "16:50", "price": 3100, "class": "economy"},
    {"flight": "SG-401", "airline": "SpiceJet", "from": "Mumbai", "to": "Jamnagar", "depart": "18:00", "arrive": "19:30", "price": 2600, "class": "economy"},
]

MOCK_TRAINS = [
    {"train": "12009", "name": "Mumbai Shatabdi", "from": "Bangalore", "to": "Mumbai", "depart": "06:00", "arrive": "17:30", "price_sleeper": 700, "price_3ac": 1800, "price_2ac": 2600},
    {"train": "11014", "name": "Kurla Express", "from": "Bangalore", "to": "Mumbai", "depart": "22:00", "arrive": "11:30+1", "price_sleeper": 550, "price_3ac": 1500, "price_2ac": 2200},
    {"train": "12936", "name": "Saurashtra Mail", "from": "Mumbai", "to": "Jamnagar", "depart": "19:50", "arrive": "07:15+1", "price_sleeper": 450, "price_3ac": 1200, "price_2ac": 1800},
    {"train": "22956", "name": "Kutch Express", "from": "Mumbai", "to": "Jamnagar", "depart": "23:15", "arrive": "10:45+1", "price_sleeper": 400, "price_3ac": 1100, "price_2ac": 1600},
]


@mcp.tool()
def search_hotels(city: str, check_in: str, check_out: str, max_price_per_night: int = 10000,
                  sort_by: str = "price") -> str:
    """Search available hotels in a city. Returns a list of hotels with prices, ratings, and amenities.
    Args: city (e.g. 'mumbai'), check_in (YYYY-MM-DD), check_out (YYYY-MM-DD),
    max_price_per_night (budget cap), sort_by ('price'|'rating'|'distance')."""
    key = city.lower().strip()
    hotels = MOCK_HOTELS.get(key, [])
    if not hotels:
        return json.dumps({"error": f"No hotels found in '{city}'. Available cities: {list(MOCK_HOTELS.keys())}"})

    results = [h for h in hotels if h["price_per_night"] <= max_price_per_night]
    if sort_by == "rating":
        results.sort(key=lambda h: -h["rating"])
    elif sort_by == "distance" and "distance_to_jio_world_centre_km" in results[0]:
        results.sort(key=lambda h: h.get("distance_to_jio_world_centre_km", 999))
    else:
        results.sort(key=lambda h: h["price_per_night"])

    from datetime import datetime as dt
    try:
        d1 = dt.strptime(check_in, "%Y-%m-%d")
        d2 = dt.strptime(check_out, "%Y-%m-%d")
        nights = (d2 - d1).days
    except ValueError:
        nights = 1

    for h in results:
        h["check_in"] = check_in
        h["check_out"] = check_out
        h["nights"] = nights
        h["total_price"] = h["price_per_night"] * nights

    return json.dumps({"city": city, "check_in": check_in, "check_out": check_out,
                        "nights": nights, "hotels": results, "count": len(results)}, indent=2)


@mcp.tool()
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights between two cities on a given date.
    Args: origin (city name), destination (city name), date (YYYY-MM-DD)."""
    o = origin.lower().strip()
    d = destination.lower().strip()
    results = [f for f in MOCK_FLIGHTS if f["from"].lower() == o and f["to"].lower() == d]
    if not results:
        return json.dumps({"error": f"No flights from '{origin}' to '{destination}'. Try swapping or checking city names."})
    for f in results:
        f["date"] = date
    return json.dumps({"origin": origin, "destination": destination, "date": date,
                        "flights": results, "count": len(results)}, indent=2)


@mcp.tool()
def search_trains(origin: str, destination: str, date: str) -> str:
    """Search available trains between two cities on a given date.
    Args: origin (city name), destination (city name), date (YYYY-MM-DD)."""
    o = origin.lower().strip()
    d = destination.lower().strip()
    results = [t for t in MOCK_TRAINS if t["from"].lower() == o and t["to"].lower() == d]
    if not results:
        return json.dumps({"error": f"No trains from '{origin}' to '{destination}'."})
    for t in results:
        t["date"] = date
    return json.dumps({"origin": origin, "destination": destination, "date": date,
                        "trains": results, "count": len(results)}, indent=2)


@mcp.tool()
def book_hotel(hotel_name: str, city: str, check_in: str, check_out: str,
               guest_name: str = "User") -> str:
    """Book a hotel room (mock). Creates a confirmed booking record.
    Args: hotel_name, city, check_in (YYYY-MM-DD), check_out (YYYY-MM-DD), guest_name."""
    key = city.lower().strip()
    hotels = MOCK_HOTELS.get(key, [])
    match = [h for h in hotels if h["name"].lower() == hotel_name.lower().strip()]
    if not match:
        return json.dumps({"error": f"Hotel '{hotel_name}' not found in {city}."})

    hotel = match[0]
    from datetime import datetime as dt
    try:
        nights = (dt.strptime(check_out, "%Y-%m-%d") - dt.strptime(check_in, "%Y-%m-%d")).days
    except ValueError:
        nights = 1

    import random
    booking = {
        "booking_id": f"HTL-{random.randint(100000, 999999)}",
        "status": "CONFIRMED",
        "hotel": hotel["name"],
        "city": city,
        "area": hotel["area"],
        "check_in": check_in,
        "check_out": check_out,
        "nights": nights,
        "price_per_night": hotel["price_per_night"],
        "total_price": hotel["price_per_night"] * nights,
        "guest": guest_name,
        "amenities": hotel["amenities"],
    }

    bookings = _load_bookings()
    bookings.append(booking)
    _save_bookings(bookings)

    return json.dumps({"message": "Booking confirmed!", "booking": booking}, indent=2)


@mcp.tool()
def book_flight(flight_number: str, date: str, passenger_name: str = "User") -> str:
    """Book a flight ticket (mock). Creates a confirmed booking record.
    Args: flight_number (e.g. 'AI-801'), date (YYYY-MM-DD), passenger_name."""
    match = [f for f in MOCK_FLIGHTS if f["flight"].lower() == flight_number.lower().strip()]
    if not match:
        return json.dumps({"error": f"Flight '{flight_number}' not found."})

    flight = match[0]
    import random
    booking = {
        "booking_id": f"FLT-{random.randint(100000, 999999)}",
        "status": "CONFIRMED",
        "flight": flight["flight"],
        "airline": flight["airline"],
        "from": flight["from"],
        "to": flight["to"],
        "date": date,
        "depart": flight["depart"],
        "arrive": flight["arrive"],
        "price": flight["price"],
        "passenger": passenger_name,
        "seat": f"{random.randint(1,30)}{random.choice('ABCDEF')}",
    }

    bookings = _load_bookings()
    bookings.append(booking)
    _save_bookings(bookings)

    return json.dumps({"message": "Flight booked!", "booking": booking}, indent=2)


@mcp.tool()
def book_train(train_number: str, date: str, travel_class: str = "3ac",
               passenger_name: str = "User") -> str:
    """Book a train ticket (mock). Creates a confirmed booking record.
    Args: train_number (e.g. '12009'), date (YYYY-MM-DD), travel_class ('sleeper'|'3ac'|'2ac'), passenger_name."""
    match = [t for t in MOCK_TRAINS if t["train"] == train_number.strip()]
    if not match:
        return json.dumps({"error": f"Train '{train_number}' not found."})

    train = match[0]
    price_key = f"price_{travel_class}"
    price = train.get(price_key, train.get("price_3ac", 0))

    import random
    booking = {
        "booking_id": f"TRN-{random.randint(100000, 999999)}",
        "pnr": f"PNR{random.randint(1000000000, 9999999999)}",
        "status": "CONFIRMED",
        "train": train["train"],
        "name": train["name"],
        "from": train["from"],
        "to": train["to"],
        "date": date,
        "depart": train["depart"],
        "arrive": train["arrive"],
        "class": travel_class,
        "price": price,
        "passenger": passenger_name,
    }

    bookings = _load_bookings()
    bookings.append(booking)
    _save_bookings(bookings)

    return json.dumps({"message": "Train ticket booked!", "booking": booking}, indent=2)


@mcp.tool()
def my_bookings() -> str:
    """View all current bookings (hotels, flights, trains)."""
    bookings = _load_bookings()
    if not bookings:
        return json.dumps({"message": "No bookings found.", "bookings": []})
    return json.dumps({"count": len(bookings), "bookings": bookings}, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
