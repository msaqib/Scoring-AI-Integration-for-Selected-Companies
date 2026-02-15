import anthropic, os, json, time

API_KEY = "" #API KEY
MODEL = "claude-3-haiku-20240307"
FOLDER = "10k_filings"
OUT = "ai_scores_intensity.jsonl"

client = anthropic.Anthropic(api_key=API_KEY)

PROMPT = """
You are an expert economist. Classify "AI Strategic Intensity" (0-5) based on the 10-K.
0-1: Generic/Risk factors.
2-3: Adopters/Internal efficiency.
4-5: Innovators/Deep Learning/NLP/Patents.
Output JSON only: { "ai_score": <int>, "reasoning": "<text>" }
"""

def run():
    if not os.path.exists(FOLDER):
        print("❌ Folder not found.")
        return

    # Check progress to avoid re-scoring
    done = set()
    if os.path.exists(OUT):
        with open(OUT, 'r') as f:
            for line in f:
                try: done.add(json.loads(line)['filename'])
                except: pass

    files = [f for f in os.listdir(FOLDER) if f.endswith("_2018.txt") and f not in done]
    files.sort()

    with open(OUT, 'a') as f_out:
        for fname in files:
            print(f"Scoring {fname}...", end=" ", flush=True)
            with open(os.path.join(FOLDER, fname), 'r') as f_in:
                # Reduced slightly to stay under token limits
                text = f_in.read()[:80000] 

            # Retry loop for Rate Limits
            while True:
                try:
                    res = client.messages.create(
                        model=MODEL, max_tokens=300, system=PROMPT,
                        messages=[{"role": "user", "content": text}]
                    )
                    f_out.write(json.dumps({"filename": fname, "response": res.content[0].text}) + "\n")
                    f_out.flush()
                    print("✅")
                    time.sleep(2) # Base pause to avoid hitting limits again
                    break 
                except anthropic.RateLimitError:
                    print("⏳ Rate limit hit. Waiting 20 seconds...")
                    time.sleep(20)
                except Exception as e:
                    print(f"❌ Error: {e}")
                    break

if __name__ == "__main__":
    run()