"""Download 10-K, Exhibit 21 and DEF 14A filings from SEC EDGAR and split them into sections."""

import json
import re
import time
from datetime import date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from kgrag.models import Document, Section

# Semiconductor / AI supply chain: dense supplier, customer, competitor and board relationships.
DEFAULT_TICKERS = [
    "NVDA",
    "AMD",
    "INTC",
    "QCOM",
    "AVGO",
    "TXN",
    "MU",
    "ADI",
    "MRVL",
    "AMAT",
    "LRCX",
    "KLAC",
    "SNPS",
    "CDNS",
    "ON",
    "MCHP",
    "SMCI",
    "DELL",
    "HPE",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AAPL",
    "ORCL",
]
TENK_ITEMS = ("1", "1A", "7")

_ITEM_RE = re.compile(
    r"^\s*item\s+(1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)\b[\s\.:\-—–]",
    re.IGNORECASE | re.MULTILINE,
)
_BIO_RE = re.compile(
    r"(director since|has served as (a|our) director|^\s*age:?\s*(\d{2})?\s*$|\bage:?\s+\d{2}\b|"
    r"other (current )?public company boards?\s*:|member of (our|the) board since|"
    r"(appointed|elected) to (our|the) board|joined (our|the) board)",
    re.IGNORECASE | re.MULTILINE,
)
FALLBACK_MIN_CHARS = 20_000
FALLBACK_CHARS = 250_000


class Edgar:
    def __init__(self, user_agent: str, min_interval: float = 0.15):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "EDGAR_USER_AGENT must be set to 'Your Name your@email.com' (SEC fair-access policy)"
            )
        self.http = httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True)
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, url: str) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.http.get(url)
        r.raise_for_status()
        return r

    def ticker_map(self) -> dict[str, dict]:
        data = self.get("https://www.sec.gov/files/company_tickers.json").json()
        return {row["ticker"]: {"cik": int(row["cik_str"]), "title": row["title"]} for row in data.values()}

    def latest_filings(self, cik: int, forms: tuple[str, ...]) -> dict[str, dict]:
        sub = self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        recent = sub["filings"]["recent"]
        found: dict[str, dict] = {}
        for i, form in enumerate(recent["form"]):
            if form in forms and form not in found:
                found[form] = {
                    "accession": recent["accessionNumber"][i],
                    "primary": recent["primaryDocument"][i],
                    "date": recent["filingDate"][i],
                }
        return found

    def filing_url(self, cik: int, accession: str, name: str) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{name}"

    def exhibit21_name(self, cik: int, accession: str) -> str | None:
        idx = self.get(self.filing_url(cik, accession, "index.json")).json()
        for item in idx["directory"]["item"]:
            name = item["name"].lower()
            if re.search(r"ex-?21", name) and name.endswith((".htm", ".html", ".txt")):
                return item["name"]
        return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_10k_items(text: str, keep: tuple[str, ...] = TENK_ITEMS) -> list[Section]:
    """Split a 10-K into Items. The table of contents repeats every heading, so for each Item keep
    the longest span between consecutive headings (the real section, not the TOC line)."""
    matches = list(_ITEM_RE.finditer(text))
    # Runs of the same Item (page running headers like "PART I / Item 1") form one span that ends at the
    # next heading of a different Item.
    spans: list[tuple[str, int, int]] = []
    for m in matches:
        key = m.group(1).upper()
        if spans and spans[-1][0] == key:
            continue
        if spans:
            spans[-1] = (spans[-1][0], spans[-1][1], m.start())
        spans.append((key, m.start(), len(text)))
    best: dict[str, tuple[int, int]] = {}
    for key, start, end in spans:
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (start, end)
    sections = []
    for key in keep:
        if key in best:
            s, e = best[key]
            body = text[s:e].strip()
            if len(body) > 500:
                sections.append(Section(path=f"10-K/Item {key}", text=body))
    if sum(len(s.text) for s in sections) < FALLBACK_MIN_CHARS:
        # Some filers (e.g. Intel) use their own layout with Item labels only in a cross-reference index.
        # Business, MD&A and risk factors come first there; financial statements follow.
        return [Section(path="10-K/Narrative (no Item headings)", text=text[:FALLBACK_CHARS].strip())]
    return sections


def extract_director_bios(
    text: str, max_chars: int = 60_000, before: int = 600, after: int = 3000
) -> list[Section]:
    """Proxy statements are long and laid out inconsistently (prose paragraphs, label/value blocks, tables).
    Every layout puts the bio next to an anchor like "Director since" or "Age", so keep a window of text
    around each anchor and merge overlapping windows."""
    windows: list[list[int]] = []
    for m in _BIO_RE.finditer(text):
        s, e = max(0, m.start() - before), min(len(text), m.end() + after)
        if windows and s <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], e)
        else:
            windows.append([s, e])
    out, size = [], 0
    for s, e in windows:
        piece = text[s:e].strip()
        if size + len(piece) > max_chars:
            piece = piece[: max_chars - size]
        out.append(piece)
        size += len(piece)
        if size >= max_chars:
            break
    return [Section(path="DEF 14A/Director and officer biographies", text="\n\n".join(out))] if out else []


def _cached_text(path: Path, fetch) -> str:
    if path.exists():
        return path.read_text()
    text = fetch()
    path.write_text(text)
    return text


def download(tickers: list[str], raw_dir: Path, user_agent: str) -> list[Path]:
    edgar = Edgar(user_agent)
    tmap = edgar.ticker_map()
    text_dir = raw_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "company_tickers.json").write_text(json.dumps(tmap))
    written: list[Path] = []
    for ticker in tickers:
        info = tmap.get(ticker)
        if not info:
            print(f"[skip] {ticker}: not in EDGAR ticker map")
            continue
        cik, title = info["cik"], info["title"]
        filings = edgar.latest_filings(cik, ("10-K", "DEF 14A"))
        for form, f in filings.items():
            doc_id = f"{ticker}-{form.replace(' ', '')}-{f['date']}"
            out = raw_dir / f"{doc_id}.json"
            if out.exists():
                written.append(out)
                continue
            url = edgar.filing_url(cik, f["accession"], f["primary"])
            # full text is kept so section parsing can be re-run without re-downloading
            text = _cached_text(text_dir / f"{doc_id}.txt", lambda u=url: html_to_text(edgar.get(u).text))
            if form == "10-K":
                sections = split_10k_items(text)
                ex21_path = text_dir / f"{doc_id}.ex21.txt"
                if not ex21_path.exists() and (ex21 := edgar.exhibit21_name(cik, f["accession"])):
                    ex21_url = edgar.filing_url(cik, f["accession"], ex21)
                    _cached_text(ex21_path, lambda u=ex21_url: html_to_text(edgar.get(u).text))
                if ex21_path.exists():
                    sections.append(Section(path="10-K/Exhibit 21 Subsidiaries", text=ex21_path.read_text()))
            else:
                sections = extract_director_bios(text)
            if not sections:
                print(f"[warn] {doc_id}: no sections parsed")
                continue
            doc = Document(
                doc_id=doc_id,
                ticker=ticker,
                company_name=title,
                cik=cik,
                form=form,
                filing_date=date.fromisoformat(f["date"]),
                url=url,
                sections=sections,
            )
            out.write_text(doc.model_dump_json(indent=1))
            written.append(out)
            print(f"[ok] {doc_id}: {len(sections)} sections, {sum(len(s.text) for s in sections):,} chars")
    return written


def reparse(raw_dir: Path) -> list[Document]:
    """Re-split every downloaded filing from its cached full text (no network)."""
    docs = []
    for doc in load_documents(raw_dir):
        text_path = raw_dir / "text" / f"{doc.doc_id}.txt"
        if not text_path.exists():
            continue
        text = text_path.read_text()
        if doc.form == "10-K":
            sections = split_10k_items(text)
            ex21_path = raw_dir / "text" / f"{doc.doc_id}.ex21.txt"
            if ex21_path.exists():
                sections.append(Section(path="10-K/Exhibit 21 Subsidiaries", text=ex21_path.read_text()))
        else:
            sections = extract_director_bios(text)
        doc = doc.model_copy(update={"sections": sections})
        (raw_dir / f"{doc.doc_id}.json").write_text(doc.model_dump_json(indent=1))
        docs.append(doc)
    return docs


def load_documents(raw_dir: Path, limit: int | None = None) -> list[Document]:
    paths = sorted(p for p in raw_dir.glob("*.json") if p.name != "company_tickers.json")
    docs = [Document.model_validate_json(p.read_text()) for p in paths]
    return docs[:limit] if limit else docs


def load_ticker_map(raw_dir: Path) -> dict[str, dict]:
    p = raw_dir / "company_tickers.json"
    return json.loads(p.read_text()) if p.exists() else {}
