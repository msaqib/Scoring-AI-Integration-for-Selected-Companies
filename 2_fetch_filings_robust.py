import os
import re
import json
from collections import defaultdict
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import shutil


# Config
INPUT = "Pilot_Truth_Data.csv"
FOLDER = "10k_filings"
EMAIL = "28100250@lums.edu.pk"

KEYWORDS = re.compile(
    r"""
    \b(
        artificial\ intelligence|
        machine\ learning|
        deep\ learning|
        generative\ ai|
        genai|
        ai-powered|
        ai-driven|
        ai-based|
        ai-enabled|
        ai\ platform|
        ai\ system|
        ai\ solution|
        ai\ model|
        ai\ capability|
        ai\ technology|
        ai\ tools?|
        large\ language\ model[s]?|
        llm[s]?|
        foundation\ model[s]?|
        transformer\ model[s]?|
        neural\ network[s]?|
        neural\ net[s]?|
        computer\ vision|
        natural\ language\ processing|
        nlp\b|
        predictive\ analytics|
        advanced\ analytics|
        intelligent\ automation|
        cognitive\ computing|
        robotics?\ process\ automation|
        rpa\b|
        algorithmic\ decision|
        autonomous\ system[s]?|
        recommender\ system[s]?|
        data[-\s]?driven\ model[s]?|
        generative\ model[s]?|
        ai\b
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def extract_main_10k(raw_text):
    """
    Extract the <DOCUMENT> block where <TYPE> is 10-K
    """
    documents = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )

    for doc in documents:
        if re.search(r"<TYPE>\s*10-K", doc, re.IGNORECASE):
            match = re.search(
                r"<TEXT>(.*?)</TEXT>",
                doc,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                return match.group(1)

    return None

def normalize_text(text):
    # Replace non-breaking spaces and other Unicode spaces
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_visible(tag):
    current = tag

    while current:
        style = current.get("style", "")
        if style:
            style = style.lower()
            for decl in style.split(";"):
                if ":" in decl:
                    prop, value = decl.split(":", 1)
                    if prop.strip() == "display" and value.strip() == "none":
                        return False

        current = current.parent

    return True

def remove_hidden_elements(soup):
    hidden_tags = soup.find_all(
        lambda t: t.get("style") and
        "display" in t.get("style").lower() and
        "none" in t.get("style").lower()
    )

    for tag in hidden_tags:
        tag.decompose()


def extract_candidate_paragraphs(html):
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    remove_hidden_elements(soup)

    candidates = []
    seen = set()

    blocks = soup.find_all(["p", "div", "li"])

    for block in blocks:
        text = block.get_text(" ", strip=True)

        text = normalize_text(text)

        if len(text) < 60:
            continue

        if KEYWORDS.search(text):
            if text not in seen:
                seen.add(text)
                candidates.append(text)

    return candidates



# ==============================
# YEAR INFERENCE FROM HEADER
# ==============================

def extract_filing_year(raw_text):
    """
    Extract FILED AS OF DATE from SEC header
    """
    match = re.search(
        r"FILED AS OF DATE:\s*(\d{8})",
        raw_text
    )
    if match:
        date_str = match.group(1)
        return int(date_str[:4])
    return None


# ==============================
# MAIN PIPELINE FUNCTION
# ==============================
def collect_ai_paragraphs(
    cik_list,
    email,
    download_folder="sec_downloads/sec-edgar-filings",
    start_year=2017,
    end_year=2025
):

    os.makedirs(download_folder, exist_ok=True)
    dl = Downloader("AI_Research_Project", email, download_folder)

    results = {}

    for cik in cik_list:
        cik_str = str(cik).zfill(10)
        print(f"\nProcessing CIK {cik_str}")

        results[cik_str] = []

        dl.get(
            "10-K",
            cik_str,
            after=f"{start_year}-01-01",
            before=f"{end_year + 1}-01-01",
        )

        cik_path = os.path.join(download_folder, cik_str, "10-K")
        if not os.path.exists(cik_path):
            continue

        for accession_dir in os.listdir(cik_path):
            accession_path = os.path.join(cik_path, accession_dir)
            if not os.path.isdir(accession_path):
                continue

            full_submission_path = os.path.join(accession_path, "full-submission.txt")
            if not os.path.exists(full_submission_path):
                continue

            with open(full_submission_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()

            filing_year = extract_filing_year(raw)
            if filing_year is None:
                continue

            if filing_year < start_year or filing_year > end_year:
                continue

            html = extract_main_10k(raw)
            if not html:
                continue

            paragraphs = extract_candidate_paragraphs(html)

            # Always register the year — even if no matches
            if len(paragraphs) == 0:
                results[cik_str].append({
                    "cik": cik_str,
                    "filing_year": filing_year,
                    "text": None
                })
            else:
                for para in paragraphs:
                    results[cik_str].append({
                        "cik": cik_str,
                        "filing_year": filing_year,
                        "text": para
                    })

    return results


def save_results_per_company(results, output_folder="ai_extracted"):
    """
    Writes a transformed JSON per CIK in the same format produced by
    `2b_convert_json.py`.  Each file will look like:

        {
            "cik": "0000001234",
            "filings": {
                "2018": [ {"text": "..."}, ... ],
                "2019": [...],
                ...
            }
        }

    This eliminates the need for the separate conversion step.
    """

    os.makedirs(output_folder, exist_ok=True)

    for cik, records in results.items():
        grouped = defaultdict(list)
        for rec in records:
            year = str(rec["filing_year"])
            if rec["text"] is not None:
                grouped[year].append({"text": rec["text"]})
            else:
                # ensure year exists but don't add fake paragraph
                grouped.setdefault(year, [])
        output_data = {
            "cik": cik,
            "filings": dict(grouped)
        }

        output_path = os.path.join(output_folder, f"{cik}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        total = sum(len(v) for v in grouped.values())
        print(f"Saved {total} records grouped by year → {output_path}")


def download():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    dl = Downloader("LUMS_Research", EMAIL, os.path.abspath(FOLDER))
    
    df = pd.read_csv(INPUT)
    print("Starting downloads...")

    data = collect_ai_paragraphs(
    cik_list=df['cik'],
    email=EMAIL
    )

    print(json.dumps(data["0000320193"][:2], indent=2))
    save_results_per_company(data)


if __name__ == "__main__":
    # existing call for full-data run:
    download()

    # quick‑check block – uncomment when you want to smoke‑test
    # small_ciks = ["0000006281", "0000050863"]   # Apple + Intel, for instance
    # data = collect_ai_paragraphs(cik_list=small_ciks, email=EMAIL,
    #                             start_year=2023, end_year=2024)
    # save_results_per_company(data, output_folder="ai_extracted_test")