import pandas as pd
import json
import os
import glob

def build_real_excel():
    jsonl_files = glob.glob("*.jsonl") + glob.glob("llm_outputs/*.jsonl")
    
    if not jsonl_files:
        print("No .jsonl files found in the main folder or llm_outputs/!")
        return

    records = []
    print(f"Reading these files: {jsonl_files}")

    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    cik = str(data.get("cik")).replace('.0', '').strip().zfill(10)
                    year = data.get("year")
                    
                    parsed = data.get("response_parsed", {})
                    
                    # Skip error rows so they don't overwrite good data with 0s
                    if not isinstance(parsed, dict) or "error" in parsed or "exception" in parsed:
                        continue
                        
                    ai_score = float(parsed.get("ai_relevance_score", 0))
                    sig_score = float(parsed.get("significance_score", 0))
                    
                    row = {
                        "cik": cik,
                        "year": int(year) if year else 0,
                        "ai_relevance_sum": ai_score,
                        "significance_sum": sig_score
                    }
                    
                    for category in ["direction", "topics", "timeline", "aggressiveness"]:
                        cat_data = parsed.get(category, {})
                        if isinstance(cat_data, dict):
                            for key, val in cat_data.items():
                                row[f"{category}_{key}"] = float(val) * ai_score
                                
                    records.append(row)
                except Exception:
                    continue

    if not records:
        print("No valid data found in any of the files!")
        return

    df = pd.DataFrame(records)
    
    #Attempt to keep the HIGHEST score for a CIK/Year (cleans up any old error overlaps) - did not work (still have 0.0 for many companies)
    print("Cleaning and aggregating data...")
    agg_df = df.groupby(['cik', 'year']).max().reset_index()
    
    #Matching company names
    try:
        master_file = "company_ciks.csv"
        if os.path.exists(master_file):
            cik_df = pd.read_csv(master_file)
            cik_col = next((col for col in cik_df.columns if 'cik' in str(col).lower()), cik_df.columns[0])
            name_col = next((col for col in cik_df.columns if 'name' in str(col).lower() or 'conm' in str(col).lower()), cik_df.columns[1])

            cik_df[cik_col] = cik_df[cik_col].astype(str).str.replace('.0', '', regex=False).str.strip().str.zfill(10)
            mapping = cik_df[[cik_col, name_col]].drop_duplicates()
            mapping.rename(columns={cik_col: 'cik', name_col: 'company_name'}, inplace=True)
            
            agg_df = agg_df.merge(mapping, on='cik', how='left')
            
            cols = agg_df.columns.tolist()
            if 'company_name' in cols:
                cols.insert(1, cols.pop(cols.index('company_name')))
                agg_df = agg_df[cols]
        else:
            print(f"Warning: {master_file} missing.")
            
    except Exception as e:
        print(f"Name merge error: {e}")
        
    output_name = "Final_AI_Integration_Scores.xlsx"
    agg_df.to_excel(output_name, index=False)
    print(f"\nSUCCESS! Found {len(agg_df)} unique records")

if __name__ == "__main__":
    build_real_excel()