import anthropic, os, json, re, time

# --- CONFIGURATION ---
API_KEY = "" #API KEY
MODEL = "claude-3-haiku-20240307" #tried sonnet model but it didnt work so shifted back to haiku
FOLDER = "10k_filings"
OUT = "ai_exposure_results.jsonl"

client = anthropic.Anthropic(api_key=API_KEY)

# Professor's Labor Economist Prompt (O*NET Logic)
PROMPT = """
You are an expert labor economist. Estimate GenAI exposure (0-1).
1. Infer major occupational groups from the 10-K.
2. Map these groups to O*NET task categories.
3. Estimate time savings for those tasks (0.0, 0.5, 1.0).
4. Calculate a weighted firm-level exposure score.

Output Format:
... [Reasoning] ...
Final Exposure Score: 0.XX
"""

def run():
    if not os.path.exists(FOLDER): 
        print(f"❌ Error: {FOLDER} not found.")
        return

    # Resume capability (Reads existing JSONL to skip done files)
    done = set()
    if os.path.exists(OUT):
        with open(OUT, 'r') as f:
            for line in f:
                try: done.add(json.loads(line)['filename'])
                except: pass

    files = [f for f in os.listdir(FOLDER) if f.endswith("_2018.txt") and f not in done]
    files.sort()

    print(f"🚀 Starting Batch B (Labor Exposure) using {MODEL}...")
    print(f"   Files to process: {len(files)}")

    with open(OUT, 'a') as f_out:
        for fname in files:
            print(f"Analyzing {fname}...", end=" ", flush=True)
            with open(os.path.join(FOLDER, fname), 'r') as f_in:
                # Sonnet token limit safety (90k chars is safe)
                text = f_in.read()[:90000] 

            # Robust Retry Loop
            while True:
                try:
                    res = client.messages.create(
                        model=MODEL, max_tokens=1500, system=PROMPT,
                        messages=[{"role": "user", "content": text}]
                    )
                    
                    ans = res.content[0].text
                    # Extract score
                    score = re.search(r"Final Exposure Score:\s*([0-9\.]+)", ans)
                    score = float(score.group(1)) if score else None
                    
                    f_out.write(json.dumps({"filename": fname, "score": score, "reasoning": ans}) + "\n")
                    f_out.flush()
                    print("✅")
                    # Sonnet is slower/expensive, pause to be safe
                    time.sleep(2) 
                    break 
                except anthropic.RateLimitError:
                    print("⏳ Rate limit hit. Waiting 30s...")
                    time.sleep(30)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    time.sleep(5)
                    break

if __name__ == "__main__":
    run()