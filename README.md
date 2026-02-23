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

### 3. AI Scoring (The Core Analysis - Script C is the latest one)
* **Script A:** `3a_compute_intensity_old_prompt(haiku_model).py`
    * **Input:** `10k_filings/`
    * **Output:** `ai_scores_intensity.jsonl` (Raw AI intensity scores)
* **Script B:** `3b_compute_exposure_new_prompt(haiku_model).py`
    * **Input:** `10k_filings/`
    * **Output:** `ai_exposure_results.jsonl` (Raw AI exposure/risk analysis)
* **Script C:** `3c_process_extracted.py`
    * **Input:** `10k_filings/`
    * **Output:** `llm_outputs/` (A folder containing .jsonl files. `adoption_output<n>.jsonl` contains the latest aggregate AI intensity scores determine through AI. Here `<n>` is a three digit decimal number which is auto incremented in each run. `output.jsonl` contains a detailed log of all runs.)
### 4. Validation & Reporting
* **Script A:** `4a_validate_intensity.py`
    * **Input:** Pass a file path such as `llm_outputs/adoption_output001.jsonl` as command line `aifile` argument.
    * **Output:** `Batch_A_Report.xlsx` (Formatted Excel report for Intensity)
    * **Output**: `residuals.png` Stores the residuals after comparison of the AI scores against the ground truth. This can be used to see which firms the AI is messing up most on.
* **Script B:** `4b_validate_exposure.py`
    * **Input:** `ai_exposure_results.jsonl`
    * **Output:** `Batch_B_Exposure_Report.xlsx` (Formatted Excel report for Exposure)
 
Sonnet Model which was used previously is not working for some reason this time.
