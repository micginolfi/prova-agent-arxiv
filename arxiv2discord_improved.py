import os, re, textwrap, subprocess, requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup

ARXIV_NEW_URL = "https://arxiv.org/list/astro-ph.GA/new"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
LLAMA_BIN = os.getenv("LLAMA_BIN", "./llama.cpp/build/bin/llama-cli")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
SEED = int(os.getenv("SEED", "42"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))
CTX = int(os.getenv("CTX", "8192"))

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

# Prompt ottimizzato per Qwen2.5-7B (più capace di seguire istruzioni complesse)
SYSTEM_PROMPT = """You are an expert astrophysicist specializing in galaxy evolution and AGN physics. 
You analyze arXiv papers daily and provide technical summaries for researchers.

Your task: Review today's astro-ph.GA submissions and identify papers relevant to:
- Active galactic nuclei (AGN) and black hole physics
- Galaxy evolution and AGN-galaxy coevolution  
- Emission line analysis and photoionization modeling
- Metallicity measurements and chemical evolution
- Feedback processes and outflows

Write in clear, technical English for expert readers. Be concise but informative."""

USER_TEMPLATE = """Today's arXiv astro-ph.GA submissions ({date_utc}):

{paper_block}

Please analyze these papers and provide:

1. For each RELEVANT paper, write exactly this format:
**Paper: [Title]**
Authors: [First author et al.]
Link: [arXiv URL]
Summary: [2-3 sentences describing methods, key results, and significance for galaxy evolution/AGN research]

2. After all papers, add:
**Daily Overview:**
[Brief synthesis of today's trends and notable findings across all submissions]

Focus only on papers directly relevant to galaxy evolution, AGN physics, or emission line studies. Skip papers on stellar physics, cosmology, or unrelated topics."""

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
            # Estrai solo il primo autore per semplicità
            first_author = authors.split(",")[0].strip() if authors else "Unknown"
            
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
                "authors": authors, "first_author": first_author, "link": link, 
                "primary_cat": primary_cat, "abstract": abstract
            })
    return items

def filter_candidates(items):
    """Pre-filtra candidati rilevanti"""
    candidates = []
    for it in items:
        cat_ok = (it.get("primary_cat","astro-ph") in CATEGORIES_OK) or ("astro-ph" in it.get("primary_cat",""))
        if not cat_ok:
            continue
            
        text = f"{it.get('title','')} {it.get('abstract','')}"
        kw_ok = bool(KEYWORDS_RE.search(text))
        
        if kw_ok:
            candidates.append(it)
    
    # Limita a massimo 8 paper per non saturare il context
    return candidates[:8]

def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit(" ", 1)[0] + "..."

def build_paper_block(candidates):
    """Costruisce il blocco input per l'LLM"""
    lines = []
    for i, p in enumerate(candidates, 1):
        title = _truncate(p.get("title", ""), 150)
        first_author = p.get("first_author", "")
        abstract = _truncate(p.get("abstract", ""), 500)
        link = p.get("link", "")
        
        lines.append(f"{i}. Title: {title}")
        lines.append(f"   Authors: {first_author} et al.")
        lines.append(f"   Link: {link}")
        lines.append(f"   Abstract: {abstract}")
        lines.append("")
    
    block = "\n".join(lines)
    print(f"Input size: {len(block)} chars (~{len(block)//4} tokens)")
    return block

def run_llama(system_prompt: str, user_prompt: str) -> str:
    """Esegue llama.cpp con prompt ottimizzato per Qwen2.5"""
    if not os.path.exists(LLAMA_BIN):
        raise RuntimeError(f"llama.cpp binary not found at {LLAMA_BIN}")
    
    if not os.path.exists(LLM_MODEL_PATH):
        raise RuntimeError(f"Model file not found at {LLM_MODEL_PATH}")

    # Usa il formato chat template di Qwen2.5
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    cmd = [
        LLAMA_BIN, "-m", LLM_MODEL_PATH,
        "--prompt", full_prompt,
        "-n", str(MAX_TOKENS),
        "-c", str(CTX),
        "--seed", str(SEED),
        "--temp", "0.3",  # Temperatura leggermente più alta per creatività
        "--repeat-penalty", "1.1",
        "--no-display-prompt",
        "-ngl", "0",  # Nessuna GPU acceleration (CPU only)
    ]

    print(f"Running: {LLAMA_BIN} with model {os.path.basename(LLM_MODEL_PATH)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-300:]}")
            print(f"STDOUT preview: {result.stdout[:200]}")
            raise RuntimeError(f"llama-cli failed with code {result.returncode}")
        
        output = result.stdout.strip()
        if not output:
            raise RuntimeError("Empty output from llama-cli")
        
        # Rimuovi eventuali token di fine
        output = re.sub(r'<\|im_end\|>.*', '', output, flags=re.DOTALL)
        return output.strip()
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("llama-cli timed out after 300 seconds")

def clean_output(text: str) -> str:
    """Pulisce l'output del modello"""
    # Rimuovi prompt artifacts
    text = re.sub(r'^.*?(\*\*Paper:|Daily Overview:)', r'\1', text, flags=re.DOTALL)
    
    # Rimuovi ripetizioni del prompt
    for remove in ["Today's arXiv", "Please analyze", "Focus only on"]:
        if remove in text:
            idx = text.find(remove)
            text = text[:idx]
    
    return text.strip()

def fallback_list(candidates):
    """Lista di fallback se l'LLM fallisce"""
    lines = ["⚠️ LLM processing failed. Today's relevant astro-ph papers:\n"]
    for p in candidates:
        title = _truncate(p.get("title", ""), 100)
        lines.append(f"• {title}")
        lines.append(f"  {p.get('link', '')}\n")
    return "\n".join(lines)

def post_discord(text: str):
    """Posta su Discord dividendo in chunks se necessario"""
    if not DISCORD_WEBHOOK:
        print("[WARN] DISCORD_WEBHOOK not set.")
        return
    
    # Dividi in chunks da max 1900 caratteri
    chunks = []
    lines = text.split('\n')
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 > 1900:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line
    
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Sending {len(chunks)} Discord chunks")
    
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            prefix = f"**Part {i+1}/{len(chunks)}**\n" if i == 0 else f"**Part {i+1}/{len(chunks)} (cont.)**\n"
            chunk = prefix + chunk
        
        payload = {"content": chunk, "flags": 4096}  # SUPPRESS_EMBEDS
        response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=20)
        
        if not response.ok:
            print(f"Discord POST failed: {response.status_code} - {response.text[:100]}")

def main():
    print("Starting arXiv analysis...")
    
    # Fetch e parse
    html = fetch_html()
    items = parse_arxiv(html)
    candidates = filter_candidates(items)
    
    if not candidates:
        post_discord("No relevant astro-ph papers found today.")
        return
    
    print(f"Found {len(candidates)} relevant papers out of {len(items)} total")
    
    # Costruisci prompt
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    paper_block = build_paper_block(candidates)
    user_prompt = USER_TEMPLATE.format(date_utc=date_utc, paper_block=paper_block)
    
    # Esegui LLM
    try:
        print("Running LLM analysis...")
        output = run_llama(SYSTEM_PROMPT, user_prompt)
        output = clean_output(output)
        
        if len(output) < 100 or "**Paper:" not in output:
            raise RuntimeError("Output too short or malformed")
        
        print(f"Generated {len(output)} characters of analysis")
        post_discord(output)
        
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        fallback = fallback_list(candidates)
        post_discord(fallback)
    
    print("Done!")

if __name__ == "__main__":
    main()
