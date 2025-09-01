import os, re, textwrap, subprocess, tempfile, requests, shutil
from datetime import datetime, timezone
from bs4 import BeautifulSoup

ARXIV_NEW_URL = "https://arxiv.org/list/astro-ph/new"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
LLAMA_BIN = os.getenv("LLAMA_BIN", "./llama.cpp/build/bin/llama-cli")  # path al binario llama.cpp
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
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

USER_TEMPLATE = """Task: Search arXiv for new submissions and replacements from the last 24h at:
https://arxiv.org/list/astro-ph/new

Filter for galaxies/AGN topics (metallicities, AGN outflows/feedback, photoionization/emission-line physics, SMBH/host).

Input (machine-compiled):
DATE_UTC: {date_utc}
WINDOW_H: 24
CANDIDATES (title | authors | arxiv_id | link | primary_cat | abstract):
{paper_block}

Output requirements:
- English only.
- Plain ASCII, explicit http links, email/RTF-friendly.
- For EACH MATCHING PAPER produce a block:

===== PAPER =====
Title: <title>
Authors: <authors>
Link: <http link to arXiv>
Summary (5–8 sentences): <methods, results, conclusions>
Relevance: <impact for galaxy evolution; AGN–galaxy coevolution; emission-line modeling>

At the end, produce:

===== DAILY SYNTHESIS =====
<One paragraph (5–8 sentences) summarizing trends across the day, including noteworthy results from non-selected candidates that may matter for the ANDES project.>

STRICT FORMAT: use exactly the headers shown (===== PAPER =====, Title:, Authors:, Link:, Summary:, Relevance:, ===== DAILY SYNTHESIS =====). No extra sections, no markdown.
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
    selected, others = [], []
    for it in items:
        cat_ok = (it["primary_cat"] in CATEGORIES_OK) or ("astro-ph" in it["primary_cat"])
        text = f"{it['title']} || {it['abstract']}"
        kw_ok = bool(KEYWORDS_RE.search(text))
        (selected if (cat_ok and kw_ok) else others).append(it)
    return selected[:40], others[:40]

def build_block(selected, others):
    def line(p):
        s = f"{p['title']} | {p['authors']} | {p['arxiv_id']} | {p['link']} | {p['primary_cat']} | {p['abstract']}"
        return s.replace("\n"," ").strip()
    return "\n".join([line(p) for p in (selected + others)])

def run_llama(system_prompt: str, user_prompt: str) -> str:
    """
    Esegue llama.cpp in CLI, passando system+user come prompt unico.
    Richiede che LLM_MODEL_PATH punti a un .gguf (open) e LLAMA_BIN sia compilato.
    """
    if not shutil.which(LLAMA_BIN):
        raise RuntimeError(f"llama.cpp binary not found at {LLAMA_BIN}")
    prompt = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write(prompt)
        tmp = f.name
    try:
        # llama-cli parametri base CPU
        cmd = [
            LLAMA_BIN,
            "-m", LLM_MODEL_PATH,
            "-p", prompt,
            "-n", str(MAX_TOKENS),
            "-s", str(SEED),
            "-c", str(CTX),
            "--seed", str(SEED),
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=1800)
            # opzionale: stampa i primi caratteri per conferma
            print("LLAMA raw output (first 80 chars):", out[:80].replace("\n"," "))
            return out.strip()
        except subprocess.CalledProcessError as e:
            print("LLAMA failed, output was:\n", e.output)
            raise

        return out.strip()
    finally:
        try: os.unlink(tmp)
        except: pass

def post_discord(text: str):
    if not DISCORD_WEBHOOK:
        print("[WARN] DISCORD_WEBHOOK not set.")
        return
    for chunk in textwrap.wrap(text, 1800, replace_whitespace=False, drop_whitespace=False):
        requests.post(DISCORD_WEBHOOK, json={"content": chunk}, timeout=20)

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
    post_discord(txt)
    print("Done.")

if __name__ == "__main__":
    main()
