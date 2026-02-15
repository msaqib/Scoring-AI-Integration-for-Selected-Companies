"""
STEP 4a: VALIDATE STRATEGIC INTENSITY (BATCH A)
-----------------------------------------------
1. Reads AI scores from 'ai_scores_intensity.jsonl'
2. Merges with Ground Truth from 'Pilot_Truth_Data.csv'
3. Saves Excel Report & Correlation Stats.
"""
import pandas as pd
import json
import re
import os
from scipy.stats import pearsonr, spearmanr

# Files
AI_FILE = "ai_scores_intensity.jsonl"
TRUTH_FILE = "Pilot_Truth_Data.csv"
REPORT_FILE = "Batch_A_Report.xlsx"
STATS_FILE = "Batch_A_Stats.txt"

def validate_a():
    print("📊 Processing Batch A (Intensity)...")
    
    # 1. Load AI Results
    if not os.path.exists(AI_FILE):
        print(f"❌ Error: {AI_FILE} not found. Run 3a first.")
        return

    ai_data = []
    with open(AI_FILE, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                # Extract CIK from filename (e.g., "12345_2018.txt" -> 12345)
                cik = int(rec['filename'].split('_')[0])
                
                # Robust extraction of "ai_score": 5
                match = re.search(r'"ai_score"\s*:\s*(\d)', rec['response'])
                if match:
                    score = int(match.group(1))
                    ai_data.append({'cik': cik, 'ai_intensity_score': score, 'reasoning': rec['response']})
            except: pass

    if not ai_data:
        print("❌ No valid scores found in JSONL file.")
        return

    df_ai = pd.DataFrame(ai_data).drop_duplicates(subset='cik', keep='last')

    # 2. Merge with Truth
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: {TRUTH_FILE} not found. Run Step 1 first.")
        return

    df_truth = pd.read_csv(TRUTH_FILE)
    merged = pd.merge(df_ai, df_truth, on='cik', how='inner')

    # 3. Save Excel Report
    merged.to_excel(REPORT_FILE, index=False)
    print(f"✅ Excel Report saved: {REPORT_FILE} (N={len(merged)})")

    # 4. Calculate Correlations
    if len(merged) > 2:
        p_corr, p_val = pearsonr(merged['ai_intensity_score'], merged['truth_score'])
        s_corr, s_val = spearmanr(merged['ai_intensity_score'], merged['truth_score'])
        
        output_str = (
            f"BATCH A VALIDATION RESULTS\n"
            f"--------------------------\n"
            f"Sample Size: {len(merged)}\n"
            f"Pearson Correlation (Linear):  {p_corr:.4f} (p={p_val:.4f})\n"
            f"Spearman Correlation (Rank):   {s_corr:.4f} (p={s_val:.4f})\n"
            f"--------------------------\n"
            f"Target Baseline: ~0.586\n"
        )
        
        print("\n" + output_str)
        
        # Save Stats to File
        with open(STATS_FILE, "w") as f:
            f.write(output_str)
        print(f"📄 Stats saved to: {STATS_FILE}")

    else:
        print("⚠️ Not enough data for correlation.")

if __name__ == "__main__":
    validate_a()