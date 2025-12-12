################################################
# Getting Project Root and Data Directory
################################################

from pathlib import Path
import pandas as pd

# Get project root directory (relative to this script)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

data=pd.read_csv(DATA_DIR / 'AutoInsurance.csv')
data.head()

################################################
# Try Catch Exception Handling
################################################

try:
    result = 10/0
    print("Result is:", result)

except Exception as e:
    print(f"An Error Occured: {e}")
    raise e

################################################
# 
################################################