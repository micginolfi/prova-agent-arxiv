# FILE: main.py (versione corretta e definitiva)

import os
import re
import subprocess
import requests
import logging
from datetime import datetime, timezone
from bs4 import BeautifulSoup

# --- Configurazione Globale ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# URL e Webhook
ARXIV_URL = "https://arxiv.org/list/astro-ph.GA/new"
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

# Configurazione LLM da variabili d'ambiente
LLAMA_BIN = os.getenv("LLAMA_BIN", "./llama.cpp/build/bin/llama-cli")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
SEED = int(os.getenv("SEED", "42"))
MAX_TOKENS_SUMMARY = int(os.getenv("MAX_TOKENS_SUMMARY", "2048"))
CTX_SIZE = int(os.getenv("CTX_SIZE", "8192"))

# --- Interessi scientifici per la pre-selezione ---
SCIENTIFIC_INTERESTS = """
My core research interests are:
- Active Galactic Nuclei (AGN) physics, including accretion, feedback, and outflows.
- The co-evolution of supermassive black holes (SMBHs) and their host galaxies.
- Galaxy evolution, particularly galaxy mergers, morphology, and quenching processes.
- Emission line diagnostics (e.g., BPT diagrams), photoionization modeling, and metallicity studies.
- Connections between AGN activity and star formation in galaxies.
"""

# --- Prompt Templates ---
SELECTION_PROMPT_TEMPLATE = f"""
You are an expert astrophysicist assistant. Your task is to select the 5 most relevant paper titles from a list based on specific research interests.

<Interests>
{SCIENTIFIC_INTERESTS}
</Interests>

Here is the list of today's new paper titles from arXiv astro-ph.GA:
<Titles>
{{paper_list}}
</Titles>

Task: Identify the 5 titles that are most semantically aligned with the interests described above.
Respond with ONLY the paper numbers, separated by commas (e.g., "1, 3, 5, 7, 9"). Do not add any other text or explanation.
"""

SUMMARY_PROMPT_TEMPLATE = """
You are an expert astrophysicist. Your task is to provide concise, technical summaries of selected arXiv papers for a research group.

Here are the selected papers for today ({date_utc}):
{paper_block}

Your task is twofold:

1.  For EACH of the {paper_count} papers provided, write a summary in exactly this format:
    **Paper: [Title]**
    Authors: [First Author et al.]
    Link: [arXiv URL]
    Summary: [A concise 2-3 sentence summary of the abstract. Focus on methods, key results, and significance for galaxy evolution or AGN research.]

2.  After summarizing all papers, add a final overview paragraph in this format:
    **Daily Highlights:**
    [A brief synthesis of today's key findings. Connect the themes of the selected papers if possible. What are the most notable results or trends today?]
"""

# --- Funzioni Principali ---

def fetch_and_parse_arxiv():
    """
    Esegue lo scraping della pagina arXiv e estrae i dati dei paper,
    ignorando cross-listing e replacement.
    """
    logging.info(f"1. Fetching data from {ARXIV_URL}...")
    try:
        response = requests.get(ARXIV_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch arXiv page: {e}")
        return []

    logging.info("Parsing HTML content...")
    soup = BeautifulSoup(response.text, "lxml")
    
    # --- INIZIO NUOVA LOGICA DI PARSING ---
    # Basata sulla struttura HTML reale fornita.
    
    # 1. Trova il contenitore principale <dl> che ha l'ID 'articles'.
    articles_dl = soup.find('dl', id='articles')
    if not articles_dl:
        logging.warning("Could not find the main articles list (<dl id='articles'>).")
        return []

    # 2. All'interno del contenitore, trova l'header H3 per le "New submissions".
    new_submissions_header = articles_dl.find('h3', string=re.compile(r'New submissions'))
    if not new_submissions_header:
        logging.warning("Could not find the 'New submissions' header inside the articles list.")
        return []

    papers = []
    # 3. Itera su tutti gli elementi *dopo* l'header H3, all'interno dello stesso genitore.
    for element in new_submissions_header.find_next_siblings():
        # 4. Se incontriamo un altro H3, significa che la sezione "New" è finita.
        if element.name == 'h3':
            break
        
        # 5. Se l'elemento è un <dt>, è l'inizio di un paper.
        if element.name == 'dt':
            # Il <dd> con i dettagli è il suo immediato fratello.
            dd = element.find_next_sibling('dd')
            if not dd:
                continue

            link_tag = element.select_one("a[href*='/abs/']")
            title_tag = dd.select_one("div.list-title")
            authors_tag = dd.select_one("div.list-authors")
            abstract_tag = dd.select_one("p.mathjax")

            if not all([link_tag, title_tag, authors_tag, abstract_tag]):
                continue

            # Estrai il primo autore
            authors_text = authors_tag.get_text(" ", strip=True).replace("Authors:", "").strip()
            first_author = authors_text.split(",")[0].strip() if authors_text else "N/A"

            papers.append({
                "title": title_tag.get_text(" ", strip=True).replace("Title: ", ""),
                "first_author": first_author,
                "link": "https://arxiv.org" + link_tag.get("href"),
                "abstract": abstract_tag.get_text(" ", strip=True).strip()
            })
    # --- FINE NUOVA LOGICA DI PARSING ---
    
    logging.info(f"Found {len(papers)} new papers.")
    return papers


def select_papers_with_llm(papers):
    """
    Usa l'LLM per selezionare i 5 paper più rilevanti basandosi sui titoli.
    """
    logging.info("2. Starting LLM pre-selection based on titles...")
    if len(papers) <= 5:
        logging.info("Fewer than 6 papers found, selecting all of them.")
        return papers

    titles_list_str = "\n".join([f"{i+1}. {p['title']}" for i, p in enumerate(papers)])
    prompt = SELECTION_PROMPT_TEMPLATE.format(paper_list=titles_list_str)
    
    full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    cmd = [
        LLAMA_BIN, "-m", LLM_MODEL_PATH,
        "-p", full_prompt, "-n", "30", "-c", "4096",
        "--seed", str(SEED), "--temp", "0.1", "--no-display-prompt"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        output = result.stdout.strip()
        
        numbers = re.findall(r'\b(\d+)\b', output)
        selected_indices = {int(n) - 1 for n in numbers if n.isdigit() and 0 < int(n) <= len(papers)}
        
        if len(selected_indices) < 3:
            logging.warning(f"LLM selection returned too few valid indices ({len(selected_indices)}). Output: '{output}'")
            raise ValueError("LLM selection failed.")

        selected_papers = [papers[i] for i in sorted(list(selected_indices))[:5]]
        logging.info(f"LLM selected {len(selected_papers)} papers: {[p['title'][:50]+'...' for p in selected_papers]}")
        return selected_papers

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logging.error(f"LLM selection failed: {e}. Falling back to selecting the first 5 papers.")
        return papers[:5]

def summarize_papers_and_generate_overview(papers):
    """
    Prende i paper selezionati, crea un prompt unico con i loro abstract
    e chiede all'LLM di sintetizzarli e creare un overview.
    """
    logging.info("3. Starting LLM summarization for selected papers...")
    
    paper_block_lines = []
    for i, p in enumerate(papers, 1):
        paper_block_lines.append(f"--- Paper {i} ---\nTitle: {p['title']}\nAuthors: {p['first_author']} et al.\nLink: {p['link']}\nAbstract: {p['abstract']}\n")
    paper_block = "\n".join(paper_block_lines)

    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = SUMMARY_PROMPT_TEMPLATE.format(date_utc=date_utc, paper_block=paper_block, paper_count=len(papers))

    full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    cmd = [
        LLAMA_BIN, "-m", LLM_MODEL_PATH,
        "-p", full_prompt, "-n", str(MAX_TOKENS_SUMMARY), "-c", str(CTX_SIZE),
        "--seed", str(SEED), "--temp", "0.4", "--repeat-penalty", "1.1", "--no-display-prompt"
    ]

    try:
        logging.info(f"Running summarization with {os.path.basename(LLM_MODEL_PATH)}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=True)
        output = re.sub(r'<\|im_end\|>.*', '', result.stdout.strip(), flags=re.DOTALL).strip()

        if not output or "**Paper:" not in output:
            raise ValueError("LLM output is empty or malformed.")
            
        logging.info(f"Successfully generated summary of {len(output)} characters.")
        return output

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logging.error(f"LLM summarization failed: {e}")
        return None

def build_fallback_message(papers):
    """Crea un messaggio di fallback se la sintesi LLM fallisce."""
    lines = [f"⚠️ **LLM Summarization Failed!**\n\nHere are the selected papers for today:\n"]
    for p in papers:
        lines.append(f"**- {p['title']}**")
        lines.append(f"  *by {p['first_author']} et al.*")
        lines.append(f"  <{p['link']}>\n")
    return "\n".join(lines)

def post_to_discord(content):
    """Invia il contenuto a Discord, gestendo i messaggi lunghi."""
    if not DISCORD_WEBHOOK:
        logging.warning("DISCORD_WEBHOOK not set. Skipping post.")
        print("\n--- BEGIN DISCORD CONTENT ---\n" + content + "\n--- END DISCORD CONTENT ---\n")
        return

    logging.info("4. Posting message to Discord...")
    header = f"### 🔭 Astro-ph.GA Daily Briefing ({datetime.now(timezone.utc).strftime('%d %b %Y')})\n\n"
    content = header + content
    
    max_len = 1950
    chunks = []
    
    if len(content) <= max_len:
        chunks.append(content)
    else:
        current_chunk = ""
        for part in content.split('\n\n'):
            if len(current_chunk) + len(part) + 2 > max_len:
                chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += "\n\n" + part if current_chunk else part
        chunks.append(current_chunk)

    for i, chunk in enumerate(chunks):
        payload = {"content": chunk}
        try:
            response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=20)
            response.raise_for_status()
            logging.info(f"Posted chunk {i+1}/{len(chunks)} to Discord successfully.")
        except requests.RequestException as e:
            logging.error(f"Failed to post chunk {i+1} to Discord: {e}")
            if response:
                logging.error(f"Response: {response.text}")
            break

def main():
    """Flusso di lavoro principale."""
    if not all([DISCORD_WEBHOOK, LLAMA_BIN, LLM_MODEL_PATH]):
        logging.error("Missing critical environment variables. Exiting.")
        return
    if not os.path.exists(LLAMA_BIN) or not os.path.exists(LLM_MODEL_PATH):
        logging.error("LLM binary or model file not found. Exiting.")
        return

    all_papers = fetch_and_parse_arxiv()
    if not all_papers:
        post_to_discord("No new papers found on astro-ph.GA today.")
        return

    selected_papers = select_papers_with_llm(all_papers)
    if not selected_papers:
        post_to_discord("Paper selection failed and no fallback was possible.")
        return

    final_message = summarize_papers_and_generate_overview(selected_papers)
    if not final_message:
        final_message = build_fallback_message(selected_papers)

    post_to_discord(final_message)
    logging.info("Workflow completed successfully.")

if __name__ == "__main__":
    main()
