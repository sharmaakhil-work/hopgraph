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
    r"(director since|has served (as a|on)|serves on the board|board of directors of|"
    r"previously served|prior to joining|age \d{2})",
    re.IGNORECASE,
)


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
    best: dict[str, tuple[int, int]] = {}
    for m, nxt in zip(matches, matches[1:] + [None], strict=False):
        key = m.group(1).upper()
        start, end = m.start(), (nxt.start() if nxt else len(text))
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (start, end)
    sections = []
    for key in keep:
        if key in best:
            s, e = best[key]
            body = text[s:e].strip()
            if len(body) > 500:
                sections.append(Section(path=f"10-K/Item {key}", text=body))
    return sections


def extract_director_bios(text: str, max_chars: int = 40_000) -> list[Section]:
    """Proxy statements are long; keep only paragraphs that read like director/officer biographies."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    bios = [p for p in paras if _BIO_RE.search(p)]
    out, size = [], 0
    for p in bios:
        if size + len(p) > max_chars:
            break
        out.append(p)
        size += len(p)
    return [Section(path="DEF 14A/Director and officer biographies", text="\n\n".join(out))] if out else []


def download(tickers: list[str], raw_dir: Path, user_agent: str) -> list[Path]:
    edgar = Edgar(user_agent)
    tmap = edgar.ticker_map()
    raw_dir.mkdir(parents=True, exist_ok=True)
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
            text = html_to_text(edgar.get(url).text)
            if form == "10-K":
                sections = split_10k_items(text)
                ex21 = edgar.exhibit21_name(cik, f["accession"])
                if ex21:
                    ex_text = html_to_text(edgar.get(edgar.filing_url(cik, f["accession"], ex21)).text)
                    sections.append(Section(path="10-K/Exhibit 21 Subsidiaries", text=ex_text))
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


def load_documents(raw_dir: Path, limit: int | None = None) -> list[Document]:
    paths = sorted(p for p in raw_dir.glob("*.json") if p.name != "company_tickers.json")
    docs = [Document.model_validate_json(p.read_text()) for p in paths]
    return docs[:limit] if limit else docs


def load_ticker_map(raw_dir: Path) -> dict[str, dict]:
    p = raw_dir / "company_tickers.json"
    return json.loads(p.read_text()) if p.exists() else {}
