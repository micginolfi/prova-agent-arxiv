import os, re, textwrap, subprocess, tempfile, requests, shutil
from datetime import datetime, timezone
from bs4 import BeautifulSoup

ARXIV_NEW_URL = "https://arxiv.org/list/astro-ph.GA/new"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
LLAMA_BIN = os.getenv("LLAMA_BIN", "./llama.cpp/build/bin/llama-cli")  # path al binario llama.cpp
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
SEED = int(os.getenv("SEED", "42"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))  # output max
CTX = int(os.getenv("CTX", "4096"))

# Filtri: categorie e parole chiave (case-insensitive)
CATEGORIES_OK = {"astro-ph.GA", "astro-ph.CO", "astro-ph.HE", "astro-ph"}
KEYWORDS = [
    r"\bAGN\b", r"active galactic nucleus", r"outflow", r"feedback",
    r"photoionization", r"ionization parameter", r"cloudy", r"mappings",
    r"emission[- ]?line", r"\bBPT\b", r"metallicit", r"oxygen abundance",
    r"supermassive black hole", r"\bSMBH\b", r"black hole mass",
    r"co[- ]?evolution", r"narrow[- ]line region", r"\bNLR\b",
    r"broad[- ]line region", r"\bBLR\b", r"galax(y|ies)", r"quasar", r"seyfert",
]
KEYWORDS_RE = re.compile("|".join(KEYWORDS), re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are an astrophysicist assistant expert in galaxy evolution and AGN physics. "
    "Always write in English, concise but technical, for expert readers. "
    "Be strictly deterministic in structure and wording. Avoid flowery language. "
    "When uncertain, say so briefly."
)

USER_TEMPLATE = """You read today's page:
https://arxiv.org/list/astro-ph.GA/new

Input:
DATE_UTC: {date_utc}
WINDOW_H: 24
CANDIDATES (title | authors | arxiv_id | link | primary_cat | abstract):
{paper_block}

Write in English, ASCII-only. For EACH relevant paper in CANDIDATES, output EXACTLY one block:

===== PAPER =====
Title: <title>
Authors: <authors>
Link: <http://arxiv link>
Summary (3–5 sentences): <methods, results, conclusions in technical style>
Relevance: <1–2 sentences on implications for galaxy evolution / AGN–galaxy coevolution / emission-line modeling>

After listing ALL relevant papers, output EXACTLY one final block:

===== DAILY SYNTHESIS =====
<5–7 sentences on today's trends across all GA submissions, including notable points from non-selected items, with any potential implications for ANDES.>

STRICT FORMAT: use only the sections above, no duplicates, no extra headers. End your message with the exact token:
<<END>>
""".strip()


def fetch_html():
    r = requests.get(ARXIV_NEW_URL, timeout=30, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def parse_arxiv(html: str):
    soup = BeautifulSoup(html, "lxml")
    items = []
    for h3 in soup.select("h3"):
        section = h3.get_text(strip=True)
        dl = h3.find_next_sibling("dl")
        if not dl: 
            continue
        for dt, dd in zip(dl.select("dt"), dl.select("dd")):
            a_id = dt.select_one("a[href*='/abs/']")
            if not a_id: 
                continue
            link = "https://arxiv.org" + a_id.get("href")
            arxiv_id = a_id.get_text(strip=True).replace("arXiv:", "")
            title_tag = dd.select_one("div.list-title")
            title = title_tag.get_text(" ", strip=True).replace("Title: ", "") if title_tag else ""
            authors_tag = dd.select_one("div.list-authors")
            authors = authors_tag.get_text(" ", strip=True).replace("Authors:", "").strip() if authors_tag else ""
            subj_tag = dd.select_one("div.list-subjects")
            primary_cat = "astro-ph"
            if subj_tag:
                sraw = subj_tag.get_text(" ", strip=True).replace("Subjects:", "").strip()
                m = re.search(r"(astro-ph\.[A-Z]{2}|astro-ph)", sraw)
                if m: primary_cat = m.group(1)
            abstract_tag = dd.select_one("p.mathjax")
            abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""
            items.append({
                "section": section, "arxiv_id": arxiv_id, "title": title,
                "authors": authors, "link": link, "primary_cat": primary_cat,
                "abstract": abstract
            })
    return items

def filter_candidates(items):
    """Seleziona paper su galaxies/AGN e separa 'others'."""
    selected, others = [], []
    for it in items:
        cat_ok = (it.get("primary_cat","astro-ph") in CATEGORIES_OK) or ("astro-ph" in it.get("primary_cat",""))
        text = f"{it.get('title','')} || {it.get('abstract','')}"
        kw_ok = bool(KEYWORDS_RE.search(text))
        (selected if (cat_ok and kw_ok) else others).append(it)

    # Limita già qui (meno roba per l’LLM)
    # tipicamente bastano 10 paper “selected” e 5 “others”
    return selected[:8], others[:4]


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0] + " …"

def build_block(selected, others, abstract_max=600, budget_chars=12000):
    """
    Costruisce il blocco CANDIDATES rispettando un budget in caratteri.
    - tronca ogni abstract
    - si ferma quando supera il budget
    """
    lines = []
    used = 0
    def add_line(p):
        nonlocal used
        title   = _truncate(p.get("title",""), 300)
        authors = _truncate(p.get("authors","").replace("\n"," "), 300)
        abstr   = _truncate(p.get("abstract","").replace("\n"," "), abstract_max)
        line = f"{title} | {authors} | {p.get('arxiv_id','')} | {p.get('link','')} | {p.get('primary_cat','astro-ph')} | {abstr}"
        if used + len(line) + 1 > budget_chars:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    for p in selected:
        if not add_line(p): break
    for p in others:
        if not add_line(p): break

    block = "\n".join(lines)
    print(f"PROMPT SIZE (chars): {len(block)}  ~tokens≈{len(block)//4}")
    return block

def run_llama(system_prompt: str, user_prompt: str) -> str:
    import shutil, subprocess, os
    LLAMA_BIN = os.getenv("LLAMA_BIN")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
    CTX = int(os.getenv("CTX", "8192"))
    SEED = int(os.getenv("SEED", "13"))

    if not shutil.which(LLAMA_BIN):
        raise RuntimeError(f"llama.cpp binary not found at {LLAMA_BIN}")

    full_prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"

    cmd = [
        LLAMA_BIN, "-m", LLM_MODEL_PATH,
        "--prompt", full_prompt,     # prompt “grezzo”
        "-n", str(MAX_TOKENS),
        "-c", str(CTX),
        "--seed", str(SEED),
        "--simple-io",
        "--no-display-prompt",
        "-no-cnv",                   # niente chat template
        "--temp", "0.2",
        "--override-stop", "<<END>>",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        print("LLAMA STDERR:", (res.stderr or "")[-500:].replace("\n"," "))
        print("LLAMA STDOUT:", (res.stdout or "")[:200].replace("\n"," "))
        raise RuntimeError(f"llama-cli exited with code {res.returncode}")
    out = (res.stdout or "").strip()
    if not out:
        print("LLAMA produced empty stdout.")
        raise RuntimeError("empty output")
    return out

def clean_model_output(txt: str) -> str:
    if "<<END>>" in txt:
        txt = txt.split("<<END>>", 1)[0]

    # tieni solo dal primo PAPER in poi
    start = txt.find("===== PAPER =====")
    if start != -1:
        txt = txt[start:]

    # se il modello ripete PAPER dopo la sintesi, tronca al termine della prima SYNTHESIS
    syn = "===== DAILY SYNTHESIS ====="
    idx = txt.find(syn)
    if idx != -1:
        # prendi dalla SYNTHESIS fino a fine del paragrafo (o fine file)
        tail = txt[idx:]
        # se per caso aggiunge di nuovo "===== PAPER =====", taglia lì
        nxt = tail.find("===== PAPER =====")
        if nxt != -1:
            tail = tail[:nxt]
        txt = txt[:idx] + tail

    # rimuovi righe fantasma dell’istruzione
    for cutter in ("STRICT FORMAT:", "After listing ALL", "Input:", "CANDIDATES"):
        if cutter in txt:
            txt = txt.replace(cutter, "")
    return txt.strip()



def fallback_list(selected):
    lines = ["Open LLM unavailable. Fallback list:\n"]
    for it in selected:
        lines.append(f"- {it['title']} ({it['link']})")
    return "\n".join(lines)


def post_discord(text: str):
    if not DISCORD_WEBHOOK:
        print("[WARN] DISCORD_WEBHOOK not set.")
        return
    chunks = [c for c in textwrap.wrap(text, 1800, replace_whitespace=False, drop_whitespace=False) if c.strip()]
    if not chunks:
        print("[WARN] Nothing to send to Discord (empty content).")
        return
    for chunk in chunks:
        r = requests.post(DISCORD_WEBHOOK, json={"content": chunk, "flags": 4096}, timeout=20)  # 4096 = SUPPRESS_EMBEDS
        if not r.ok:
            print(f"[ERR] Discord POST {r.status_code}: {r.text[:200]}")


def main():
    html = fetch_html()
    items = parse_arxiv(html)
    selected, others = filter_candidates(items)
    if not (selected or others):
        post_discord("No astro-ph candidates found today.")
        return
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = build_block(selected, others)
    user = USER_TEMPLATE.format(date_utc=date_utc, paper_block=block)
    try:

        import os, shutil
        print("DBG LLAMA_BIN:", os.getenv("LLAMA_BIN"), "exists?", os.path.exists(os.getenv("LLAMA_BIN","")))
        print("DBG LLM_MODEL_PATH:", os.getenv("LLM_MODEL_PATH"), "exists?", os.path.exists(os.getenv("LLM_MODEL_PATH","")))
        print("Selected:", len(selected), "Others:", len(others))

        txt = run_llama(SYSTEM_PROMPT, user)
        
    except Exception as e:
        lines = ["Open LLM unavailable. Fallback list:\n"]
        for it in selected:
            lines.append(f"- {it['title']} ({it['link']})")
        txt = "\n".join(lines) if lines else f"Error: {e}"
    
    txt = clean_model_output(txt)
    if not txt or "===== PAPER =====" not in txt:
        txt = fallback_list(selected)
    
    post_discord(txt)
    print("Done.")

if __name__ == "__main__":
    main()
