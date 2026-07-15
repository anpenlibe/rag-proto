#!/usr/bin/env python3
"""
Scrape studieren.univie.ac.at/en into a RAG-ready knowledge base.

Outputs (in ../data, i.e. the repo's data/ dir — refreshes the corpus IN PLACE):
  - pages/*.md            human-inspectable clean markdown per page
  - pages.jsonl           one record per page (full clean text + metadata) — canonical
  - manifest.json         run summary (counts, sections, failures)

Chunking is NOT done here: `src/rag/chunk.py` is the single chunker (driven by Config).
After re-scraping:  PYTHONPATH=src .venv/bin/python -m rag.chunk && ... -m rag.index
"""
import json, re, time, hashlib, pathlib, sys
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE = "https://studieren.univie.ac.at/en/"
SITEMAP = "https://studieren.univie.ac.at/en/sitemap/"
UA = {"User-Agent": "Mozilla/5.0 (compatible; univie-RAG-KB/1.0; educational)"}
OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
(OUT / "pages").mkdir(parents=True, exist_ok=True)
DELAY = 0.4  # polite delay between requests

# --- map a URL to one of the 9 top-level sections shown on the landing page ---
# Order matters: more specific prefixes first.
SECTION_RULES = [
    ("Resumption of studies", ["/admission/resumption-of-studies"]),
    ("Web services",          ["/web-services-uspace-ufind-moodle-webmail"]),
    ("Accessible studies",    ["/accessible-studies"]),
    ("Entrance exam",         ["/entrance-exam"]),
    ("Tuition fee",           ["/tuition-fee"]),
    ("Graduates",             ["/graduates"]),
    ("Degree programmes",     ["/degree-programmes", "/extension-curricula",
                                "/postgraduate-programmes", "/bachelordiploma-programmes"]),
    ("Admission",             ["/admission-procedure", "/admission/"]),
    ("Study organisation",    ["/study-organisation", "/welcome", "/studying-exams",
                                "/semester-planning", "/using-ai-in-your-studies",
                                "/studying-safely", "/change-of-personal-data",
                                "/studying-healthily", "/confirmations", "/abc-of-terminology",
                                "/student-life", "/pruefungsaktiv", "/minimum-number-of-credits",
                                "/registration-for-coursesexaminations"]),
]

def section_for(url):
    path = urlparse(url).path
    for name, prefixes in SECTION_RULES:
        if any(p in path for p in prefixes):
            return name
    return "Other"

# --- language detection: trust <html lang>, fall back to a stopword heuristic ---
# Some pages under /en/ actually serve German content (e.g. teacher-education);
# we keep the corpus English-only, so detect and filter these out.
GERMAN_MARKERS = set("der die das und oder fuer für mit von den dem des ein eine "
                     "im ist auch nicht sich werden bei nach ueber über sowie durch "
                     "als um zur zum sind wird".split())
ENGLISH_MARKERS = set("the and for with from this that are you your will can not also "
                      "which their there has have been of to in on at as".split())

def detect_language(soup, text):
    # Trust the page's declared <html lang>. An 'en' page that merely quotes some
    # German (programme names, terms) stays English — we never strip it out.
    html_tag = soup.find("html")
    declared = ((html_tag.get("lang") if html_tag else "") or "").strip().lower()[:2]
    if declared:
        return declared if declared in ("en", "de") else "en"
    # No declaration: fall back to a stopword heuristic, but lean English —
    # only call it German when German clearly dominates.
    words = re.findall(r"[a-zäöüß]+", text.lower())
    de = sum(w in GERMAN_MARKERS for w in words)
    en = sum(w in ENGLISH_MARKERS for w in words)
    return "de" if de > 2 * max(en, 1) else "en"

def get(url):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code == 200:
                r.encoding = r.apparent_encoding or "utf-8"
                return r.text
            if r.status_code in (404, 410):
                return None
        except requests.RequestException as e:
            if attempt == 2:
                print(f"   ! request failed {url}: {e}", file=sys.stderr)
        time.sleep(1.0 * (attempt + 1))
    return None

def collect_urls():
    """Primary list from the sitemap; kept only for the /en/ content tree."""
    html = get(SITEMAP)
    soup = BeautifulSoup(html, "lxml")
    main = soup.find(id="mainContent") or soup
    urls = []
    seen = set()
    for a in main.find_all("a", href=True):
        full = urljoin(SITEMAP, a["href"]).split("#")[0]
        p = urlparse(full)
        if p.netloc != "studieren.univie.ac.at":
            continue
        if not p.path.startswith("/en/"):
            continue
        # skip utility pages
        if any(s in p.path for s in ("/sitemap", "/imprint", "/accessibility",
                                     "/contact-form", "/quicklinks", "/news")):
            continue
        full = full.split("?")[0]  # normalise query strings
        if full not in seen:
            seen.add(full)
            urls.append(full)
    return urls

def breadcrumb(soup):
    """Extract the 'You are here' trail."""
    for ul in soup.find_all(["ul", "ol"]):
        txt = ul.get_text(" ", strip=True)
        if "You are here" in txt:
            items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
            items = [i for i in items if i and "You are here" not in i]
            return items
    return []

def extract(url, html):
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    main = soup.select_one("#mainContent .main") or soup.find(id="mainContent")
    if main is None:
        return None

    # remove noise: in-page jump menus, "to top", share, hidden helpers
    for sel in [".csc-menu", ".exclude-search", ".divider", "script", "style",
                ".skiplink", "[class*=print]"]:
        for el in main.select(sel):
            el.decompose()
    # remove "To top" anchors
    for a in main.find_all("a"):
        if a.get_text(strip=True).lower() in ("to top", "back to top"):
            a.decompose()

    crumbs = breadcrumb(soup)

    # convert to markdown, keep heading structure & tables
    body_md = md(str(main), heading_style="ATX", bullets="-", strip=["img"])
    # tidy whitespace
    body_md = re.sub(r"\n{3,}", "\n\n", body_md).strip()
    # drop lines that are just stray markdown artefacts
    lines = [ln.rstrip() for ln in body_md.splitlines()]
    body_md = "\n".join(lines).strip()

    # H1 as the clean page title if present
    h1 = main.find("h1")
    page_title = h1.get_text(" ", strip=True) if h1 else title.replace(" | Studying", "").strip()

    return {
        "url": url,
        "title": page_title,
        "html_title": title,
        "section": section_for(url),
        "language": detect_language(soup, body_md),
        "breadcrumb": crumbs,
        "text": body_md,
        "word_count": len(body_md.split()),
    }

def slugify(url):
    path = urlparse(url).path.strip("/").replace("/", "__") or "index"
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", path)

def main():
    urls = collect_urls()
    print(f"Discovered {len(urls)} content pages from sitemap")
    pages = []
    failures = []
    filtered_non_en = []
    for i, url in enumerate(urls, 1):
        html = get(url)
        if not html:
            failures.append(url); print(f"[{i}/{len(urls)}] FAIL {url}"); continue
        rec = extract(url, html)
        if not rec or rec["word_count"] < 15:
            failures.append(url); print(f"[{i}/{len(urls)}] EMPTY {url}"); continue
        if rec["language"] != "en":
            filtered_non_en.append(url)
            print(f"[{i}/{len(urls)}] SKIP non-en ({rec['language']}) {url}")
            continue
        pages.append(rec)
        (OUT / "pages" / (slugify(url) + ".md")).write_text(
            f"# {rec['title']}\n\n"
            f"*Section: {rec['section']} · Source: {url}*\n\n{rec['text']}\n",
            encoding="utf-8")
        print(f"[{i}/{len(urls)}] {rec['section']:22s} {rec['word_count']:5d}w  {rec['title'][:50]}")
        time.sleep(DELAY)

    # write pages.jsonl
    with (OUT / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            pid = hashlib.md5(p["url"].encode()).hexdigest()[:12]
            f.write(json.dumps({"id": pid, **p}, ensure_ascii=False) + "\n")

    # NOTE: chunking deliberately does NOT happen here. `src/rag/chunk.py` is the one
    # and only chunker — it reads pages.jsonl and is driven by `Config` (so chunk
    # params are part of config_hash). A second chunker here would silently overwrite
    # data/chunks.jsonl with differently-sized chunks while traces still claimed the
    # Config values. After re-scraping, run: PYTHONPATH=src python -m rag.chunk

    # section counts
    from collections import Counter
    sec_counts = Counter(p["section"] for p in pages)
    manifest = {
        "source": BASE,
        "pages_scraped": len(pages),
        "languages": {"en": len(pages)},
        "failures": failures,
        "filtered_non_en": filtered_non_en,
        "sections": dict(sec_counts),
        "total_words": sum(p["word_count"] for p in pages),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("\n=== DONE ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
