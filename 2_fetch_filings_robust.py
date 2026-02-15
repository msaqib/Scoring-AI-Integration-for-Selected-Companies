import os
import pandas as pd
from sec_edgar_downloader import Downloader
import shutil

# Config
INPUT = "Pilot_Truth_Data.csv"
FOLDER = "10k_filings"
EMAIL = "28100250@lums.edu.pk"

def download():
    if not os.path.exists(FOLDER): os.makedirs(FOLDER)
    dl = Downloader("LUMS_Research", EMAIL, os.path.abspath(FOLDER))
    
    df = pd.read_csv(INPUT)
    print("Starting downloads...")

    for cik in df['cik']:
        name = f"{cik}_2018.txt"
        path = os.path.join(FOLDER, name)
        if os.path.exists(path): 
            continue
        
        try:
            # Get 2018 10-K
            dl.get("10-K", str(cik).zfill(10), after="2018-01-01", before="2019-01-01")
            
            # Move and clean up
            base = os.path.join(FOLDER, "sec-edgar-filings", str(cik).zfill(10), "10-K")
            if os.path.exists(base):
                fid = os.listdir(base)[0]
                src = os.path.join(base, fid, "full-submission.txt")
                shutil.move(src, path)
                shutil.rmtree(os.path.join(FOLDER, "sec-edgar-filings"))
                print(f"Saved {cik}")
        except Exception as e:
            print(f"Error {cik}: {e}")

if __name__ == "__main__":
    download()