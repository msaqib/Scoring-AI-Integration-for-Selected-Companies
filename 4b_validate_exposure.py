"""
SCRIPT: 4b_validate_exposure.py
PURPOSE: Correlates the generated AI Exposure scores with the Truth Scores.
"""
import pandas as pd
import json
import os
from scipy.stats import pearsonr, spearmanr

# Files
# Note: If you used the Haiku script I gave specifically, this might be "ai_exposure_results_haiku.jsonl"
# If you just changed the model in the main script, it is "ai_exposure_results.jsonl"
AI_FILE = "ai_exposure_results.jsonl" 
TRUTH_FILE = "Pilot_Truth_Data.csv"
REPORT_FILE = "Batch_B_Exposure_Report.xlsx"

def validate():
    print("📊 Validating Batch B (Labor Exposure)...")
    
    # 1. Check for the file (handling the potential filename difference)
    if not os.path.exists(AI_FILE):
        if os.path.exists("ai_exposure_results_haiku.jsonl"):
            print("Found 'ai_exposure_results_haiku.jsonl', using that instead.")
            current_file = "ai_exposure_results_haiku.jsonl"
        else:
            print(f"❌ Error: Could not find {AI_FILE}. Make sure the filename matches.")
            return
    else:
        current_file = AI_FILE

    # 2. Load AI Results
    ai_data = []
    with open(current_file, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                cik = int(rec['filename'].split('_')[0])
                score = rec.get('score')
                
                if score is not None:
                    ai_data.append({
                        'cik': cik, 
                        'ai_exposure_score': score,
                        'reasoning': rec['reasoning'][:500] + "..." # Truncate for Excel
                    })
            except: pass

    if not ai_data:
        print("❌ No valid scores found in the file.")
        return

    df_ai = pd.DataFrame(ai_data).drop_duplicates(subset='cik', keep='last')

    # 3. Merge with Truth
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: {TRUTH_FILE} not found. Run Step 1 first.")
        return

    df_truth = pd.read_csv(TRUTH_FILE)
    merged = pd.merge(df_ai, df_truth, on='cik', how='inner')

    # 4. Save Excel Report
    merged.to_excel(REPORT_FILE, index=False)
    print(f"✅ Report saved: {REPORT_FILE} (N={len(merged)})")

    # 5. Calculate Correlations
    if len(merged) > 2:
        p_corr, p_val = pearsonr(merged['ai_exposure_score'], merged['truth_score'])
        s_corr, s_val = spearmanr(merged['ai_exposure_score'], merged['truth_score'])
        
        print("\n" + "="*30)
        print(f"🎯 BATCH B RESULTS (Method: Exposure)")
        print("="*30)
        print(f"Sample Size: {len(merged)}")
        print(f"Pearson Correlation (r): {p_corr:.4f}  (p={p_val:.4f})")
        print(f"Spearman Correlation (ρ): {s_corr:.4f}")
        print("-" * 30)
        
        # Comparison logic
        print("INTERPRETATION:")
        if p_corr > 0.4861:
            print(f"✅ SUCCESS: Beat the Batch A baseline (0.4861).")
            print(f"   Improvement: +{p_corr - 0.4861:.4f}")
        else:
            print(f"⚠️ NOTE: Did not beat Batch A (0.4861).")
            print("   (Likely due to using the smaller 'Haiku' model)")
        print("="*30)

    else:
        print("⚠️ Not enough data for correlation.")

if __name__ == "__main__":
    validate()