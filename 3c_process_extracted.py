import os
import math
import json
import anthropic
from collections import defaultdict

# directory containing the individual JSON extraction files
BASE_DIR = "ai_extracted"
# destination for LLM outputs
OUTPUT_DIR = "llm_outputs"
OUTPUT_FILE = "output.jsonl"
ADOPTION_OUTPUT_FILE = "adoption_output.jsonl"

import hashlib
from datetime import datetime, UTC

# grab API key from environment variable, leaving empty string if not set
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-3-haiku-20240307"
TEMPERATURE = 0.2

client = anthropic.Anthropic(api_key=API_KEY)

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

Return ONLY valid JSON.
Do NOT include explanations, commentary, or additional text before or after the JSON.
If you include anything other than JSON, the output will be discarded. Here's an example output JSON string:

{
"direction": {"positive":0.9,"negative":0.1,"other":0.0},
"significance_score": 0.8,
"topics": {"labor":0.5,"investment":0.5,"revenue":0.3,"competition":0.0,"M&A":0.0,"other":0.0},
"timeline": {"happened":0.7,"current":0.2,"planning":0.1,"other":0.0},
"aggressiveness": {"active":0.6,"passive":0.4,"other":0.0},
"ai_relevance_score": 0.9,
"overall_confidence": 0.3
}"""

import json

def extract_json_object(text):
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = text[start:i+1]
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    raise ValueError("Malformed JSON object")

                # Validate required keys
                required_keys = [
                    "direction",
                    "significance_score",
                    "topics",
                    "timeline",
                    "aggressiveness",
                    "ai_relevance_score",
                    "overall_confidence"
                ]
                for key in required_keys:
                    if key not in parsed:
                        raise ValueError(f"Missing key in JSON output: {key}")

                return parsed

    raise ValueError("No complete JSON object found")

import json

def query_exposure_model(paragraph: str) -> dict:
    """
    Send a paragraph to Claude and return both:
      - raw LLM output as `raw`
      - parsed JSON as `parsed` (or an error dict if parsing fails)

    The API key must be set in the environment variable `ANTHROPIC_API_KEY`.
    """
    if not API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment")

    # Send request to the model
    res = client.messages.create(
        model=MODEL,
        system=PROMPT,
        messages=[{"role": "user", "content": paragraph}],
        max_tokens=1000,
    )

    # Extract text from the Anthropic response
    if hasattr(res, "content") and res.content:
        answer_text = res.content[0].text
    else:
        answer_text = str(res)

    parsed = None
    try:
        parsed = extract_json_object(answer_text)
    except Exception as e:
        parsed = {"error": "parse_failed", "exception": str(e)}

    # Always return both raw and parsed
    return {
        "raw": answer_text,
        "parsed": parsed
    }



def process_statement(cik: str, year: str, idx: int, text: str, outf,
                      model_version="2026-01", prompt_version="v2.1") -> None:
    """Handle a single filing statement.

    Instead of printing, serialize the information to JSONL in ``outf``.
    A paragraph ID is constructed as "{year}_p{idx}" and the hash is an
    MD5 of the paragraph text.  A UTC timestamp is added.
    """

    para_id = f"{year}_p{idx}"
    para_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    timestamp = datetime.now(UTC).isoformat() + "Z"

    parsed = None
    raw_output = None
    try:
        res = query_exposure_model(text)  # your existing function
        parsed = res.get("parsed", None)
        raw_output = res.get("raw", None)
    except Exception as e:
        parsed = {"error": "query_failed", "exception": str(e)}
        raw_output = None

    record = {
        "cik": cik,
        "year": int(year) if year.isdigit() else year,
        "paragraph_id": para_id,
        "paragraph_hash": para_hash,
        "paragraph_text": text,
        "timestamp": timestamp,
        "model_name": MODEL,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "temperature": TEMPERATURE,
        "prompt_sent": PROMPT,
        "response_raw": raw_output if raw_output else parsed if parsed else None,
        "response_parsed": parsed if parsed else None
    }
    # disable ascii escaping so unicode punctuation remains readable
    outf.write(json.dumps(record, ensure_ascii=False) + "\n")
    return parsed, raw_output

def process_year(cik: str, year: str, entries: list, outf, adoption_outf=None):
    """
    Iterate through filings of a given year, process paragraphs, and optionally aggregate
    a firm-year adoption measure.
    """

    # Initialize accumulators
    agg = {
        "raw_weight_sum": 0.0,
        "weight_sum": 0.0,
        "direction": defaultdict(float),
        "topics": defaultdict(float),
        "timeline": defaultdict(float),
        "aggressiveness": defaultdict(float)
    }

    for idx, entry in enumerate(entries, start=1):
        text = entry.get("text", "")
        parsed, raw_output = process_statement(cik, year, idx, text, outf)

        if not parsed or parsed.get("error") or parsed.get("ai_relevance_score", 0) < 0.05:
            continue

        timeline_prob = parsed["timeline"].get("current", 0.0) + parsed["timeline"].get("happened", 0.0)
        active_prob = parsed["aggressiveness"].get("active", 0.0)
        weight = parsed.get("significance_score", 1.0) * timeline_prob * active_prob * parsed.get("ai_relevance_score", 1.0)

        if weight == 0:
            continue

        agg["raw_weight_sum"] += weight

        for d, v in parsed.get("direction", {}).items():
            agg["direction"][d] += v * weight
        for t, v in parsed.get("topics", {}).items():
            agg["topics"][t] += v * weight
        for t, v in parsed.get("timeline", {}).items():
            agg["timeline"][t] += v * weight
        for a, v in parsed.get("aggressiveness", {}).items():
            agg["aggressiveness"][a] += v * weight

    n_paragraphs = sum(
        1 for idx, entry in enumerate(entries, start=1)
        if (parsed := process_statement(cik, year, idx, entry.get("text", ""), outf)[0])
        and not parsed.get("error")
        and parsed.get("ai_relevance_score", 0) >= 0.05
    )

    # Apply log + per-paragraph normalization to adoption_weight_sum
    if n_paragraphs > 0:
        adoption_score = math.log1p(agg["raw_weight_sum"]) / n_paragraphs
    else:
        adoption_score = 0.0
        
    # Normalize weighted sums
    def normalize(d):
        total = sum(d.values())
        if total > 0:
            return {k: v / total for k, v in d.items()}
        else:
            return {k: 0.0 for k in d}

    normalized = {
        "cik": cik,
        "year": int(year) if year.isdigit() else year,
        # "adoption_weight_sum": agg["weight_sum"],
        "adoption_weight_sum": adoption_score,
        "raw_adoption_weight_sum": agg["raw_weight_sum"],
        "direction": normalize(agg["direction"]),
        "topics": normalize(agg["topics"]),
        "timeline": normalize(agg["timeline"]),
        "aggressiveness": normalize(agg["aggressiveness"])
    }


    if adoption_outf:
        adoption_outf.write(json.dumps(normalized, ensure_ascii=False) + "\n")



def process_file(cik: str, filings: dict, outf, adoption_outf=None) -> None:
    """Write out all statements for a single extraction record."""

    for year, entries in filings.items():
        process_year(cik, year, entries, outf, adoption_outf)


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
    adoption_outpath = os.path.join(OUTPUT_DIR, ADOPTION_OUTPUT_FILE)
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
                process_file(cik, filings, outf, adoption_outf=open(adoption_outpath, "a") if adoption_outpath else None)


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


def test_file(path: str, outf_path=None, adoption_outf_path=None) -> None:
    """Run the LLM prompt on every paragraph in a single extraction file.

    This is useful for quick connection testing without touching the larger
    dataset.  ``path`` may be an absolute filename or relative to ``BASE_DIR``.
    """
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)

    if not os.path.isfile(path):
        print(f"file not found: {path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    cik = data.get("cik", "<unknown>")
    filings = data.get("filings", {})
    if not filings:
        print(f"No filings in {path} (CIK {cik})")
        return

    if outf_path:
        os.makedirs(os.path.dirname(outf_path), exist_ok=True)
        outf = open(outf_path, "a", encoding="utf-8")
    else:
        outf = None

    if adoption_outf_path:
        os.makedirs(os.path.dirname(adoption_outf_path), exist_ok=True)
        adoption_outf = open(adoption_outf_path, "a", encoding="utf-8")
    else:
        adoption_outf = None
    process_file(cik, filings, outf, adoption_outf)

    # for year, entries in filings.items():
    #     for idx, entry in enumerate(entries, start=1):
    #         text = entry.get("text", "")
    #         parsed, raw_output = query_exposure_model(text)
    #         if outf:
    #             record = {
    #                 "cik": cik,
    #                 "year": int(year),
    #                 "paragraph_id": f"{year}_p{idx}",
    #                 "paragraph_text": text,
    #                 "response_raw": raw_output,
    #                 "response_parsed": parsed,
    #             }
    #             outf.write(json.dumps(record, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        test_file(sys.argv[2], "llm_outputs/test_output.jsonl", "llm_outputs/test_adoption_output.jsonl")
    else:
        process_dir()
