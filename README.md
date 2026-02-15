# Scoring-AI-Integration-for-Selected-Companies
## How It Works

The scripts are numbered to be run in sequential order. Here is the data flow:

### 1. Data Preparation
* **Script:** `1_prepare_pilot_data.py`
* **Input:** `Pilot_Truth_Data.csv` (Raw list of companies)
* **Output:** `Final_List_With_CIKs.xlsx` (Cleaned list with CIK numbers needed for SEC retrieval)

### 2. Fetching Filings
* **Script:** `2_fetch_filings_robust.py`
* **Input:** `Final_List_With_CIKs.xlsx`
* **Output:** `10k_filings/` (A folder containing the raw text/HTML of the 10-K filings)

### 3. AI Scoring (The Core Analysis)
* **Script A:** `3a_compute_intensity_old_prompt(haiku_model).py`
    * **Input:** `10k_filings/`
    * **Output:** `ai_scores_intensity.jsonl` (Raw AI intensity scores)
* **Script B:** `3b_compute_exposure_new_prompt(haiku_model).py`
    * **Input:** `10k_filings/`
    * **Output:** `ai_exposure_results.jsonl` (Raw AI exposure/risk analysis)

### 4. Validation & Reporting
* **Script A:** `4a_validate_intensity.py`
    * **Input:** `ai_scores_intensity.jsonl`
    * **Output:** `Batch_A_Report.xlsx` (Formatted Excel report for Intensity)
* **Script B:** `4b_validate_exposure.py`
    * **Input:** `ai_exposure_results.jsonl`
    * **Output:** `Batch_B_Exposure_Report.xlsx` (Formatted Excel report for Exposure)
 
Sonnet Model which was used previously is not working for some reason this time.
