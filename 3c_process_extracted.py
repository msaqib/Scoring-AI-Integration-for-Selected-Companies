import os
import json

# directory containing the individual JSON extraction files
BASE_DIR = "ai_extracted"
# destination for LLM outputs
OUTPUT_DIR = "llm_outputs"
OUTPUT_FILE = "output.jsonl"

import hashlib
from datetime import datetime

PROMPT = """SYSTEM PROMPT

You are analyzing corporate disclosure text from SEC 10-K filings.

Your task is to measure realized economic integration of artificial intelligence (AI) technologies in the firm.

Carefully analyze the text and return structured probabilistic outputs.

Use the definitions below strictly.

DEFINITIONS:

Direction of impact:

positive: AI improves revenue, efficiency, competitive position, or strategic advantage.

negative: AI increases cost, risk, disruption, or competitive threat.

other: neutral, descriptive, or unclear.

Significance level (economic materiality):

low: minor operational mention.

moderate: meaningful but not central to firm strategy.

high: strategically important or financially material.

Return significance as a numeric expected value between 0 and 3.

Primary economic topic:
Return probability distribution over:

labor

investment

revenue

competition

M&A

other

Timeline:
Return probability distribution over:

happened (already realized)

current (ongoing implementation)

planning (future intention)

other

Aggressiveness:
Return probability distribution over:

active (firm directly developing/deploying AI)

passive (AI embedded in third-party tools or mentioned indirectly)

other

Additionally return:

overall_confidence (0 to 1)

ai_relevance_score (0 to 1, where 0 = not meaningfully about AI)

Respond strictly in JSON."""

def query_exposure_model(paragraph: str) -> dict:
    msg = PROMPT + "\n\n" + paragraph
    #res = client.messages.create(..., system=msg, …)
    # parse/res/return whatever you need
    return { "score":0.5, "reasoning":"This is a mock response for demonstration purposes." }


def process_statement(cik: str, year: str, idx: int, text: str, outf) -> None:
    """Handle a single filing statement.

    Instead of printing, serialize the information to JSONL in ``outf``.
    A paragraph ID is constructed as "{year}_p{idx}" and the hash is an
    MD5 of the paragraph text.  A UTC timestamp is added.
    """

    para_id = f"{year}_p{idx}"
    para_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    timestamp = datetime.utcnow().isoformat() + "Z"

    record = {
        "cik": cik,
        "year": int(year) if year.isdigit() else year,
        "paragraph_id": para_id,
        "paragraph_hash": para_hash,
        "paragraph_text": text,
        "timestamp": timestamp,
    }
    # disable ascii escaping so unicode punctuation remains readable
    outf.write(json.dumps(record, ensure_ascii=False) + "\n")

def process_year(cik: str, year: str, entries: list, outf) -> None:
    """Iterate through filings of a given year and write them out."""

    for idx, entry in enumerate(entries, start=1):
        text = entry.get("text", "")
        process_statement(cik, year, idx, text, outf)

def process_file(cik: str, filings: dict, outf) -> None:
    """Write out all statements for a single extraction record."""

    for year, entries in filings.items():
        process_year(cik, year, entries, outf)


def process_dir(path: str = BASE_DIR) -> None:
    """Walk through ``path`` and display parsed filings.

    Every ``*.json`` file in the directory is opened and its top‑level
    structure assumed to be ``{"cik": ..., "filings": {...}}``.  If the
    ``filings`` object is empty, we print a message; otherwise we delegate to
    :func:`process_file` so the year/filing texts are shown.
    """

    if not os.path.isdir(path):
        print(f"directory not found: {path}")
        return

    # ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(outpath, "a") as outf:
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".json"):
                continue
            full = os.path.join(path, fname)
            try:
                with open(full, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"failed to read {fname}: {e}")
                continue

            cik = data.get("cik", "<unknown>")
            filings = data.get("filings", {})
            if not filings:
                print(f"No filings to display for CIK: {cik}")
            else:
                process_file(cik, filings, outf)


def clean_output(path: str = os.path.join(OUTPUT_DIR, OUTPUT_FILE)) -> None:
    """Rewrite an existing JSONL output with proper unicode encoding.

    This is useful if the file was generated before ``ensure_ascii=False`` was
    added and contains escaped sequences such as ``\u2019``.  Each line is
    read and re-dumped with the correct option, overwriting the file in place.
    """

    if not os.path.isfile(path):
        print(f"no file to clean: {path}")
        return

    temp_path = path + ".tmp"
    with open(path, "r") as inp, open(temp_path, "w", encoding="utf-8") as outp:
        for line in inp:
            try:
                obj = json.loads(line)
            except Exception:
                outp.write(line)
                continue
            outp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)
    print(f"cleaned {path}")


if __name__ == "__main__":
    process_dir()
