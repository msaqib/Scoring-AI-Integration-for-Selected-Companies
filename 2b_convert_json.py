import json
from collections import defaultdict
import os


def process_filings(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Input JSON is empty.")

    # Extract cik (assuming it is identical across records)
    cik = data[0]["cik"]

    # Group texts by year
    grouped = defaultdict(list)

    for record in data:
        year = str(record["filing_year"])
        grouped[year].append({
            "text": record["text"]
        })

    # Build new structure
    result = {
        "cik": cik,
        "filings": dict(grouped)
    }

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Written transformed JSON to {output_file}")


if __name__ == "__main__":
    # process all JSON files found in ai_extracted folder
    input_dir = "ai_extracted"
    output_dir = "ai_extracted_proc"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        try:
            process_filings(in_path, out_path)
        except Exception as e:
            print(f"Failed processing {in_path}: {e}")

