##############################
# Creating Log files in Python 
##############################
import logging
import pandas
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

current_directory = os.getcwd()
print("Current Directory:", current_directory)

# --- Configure Logging ---
# This setup sends INFO-level messages (and higher) to a file and ERROR-level messages to the console (stderr).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("my_ml_project.log"),
        logging.StreamHandler(sys.stdout) # You can also log to console
    ]
)

# Get the logger object
logger = logging.getLogger(__name__)

# Use the Logger in Your Script ---
logger.info("Script started.")

# Logger information
params = {'test_size': 0.2, 'random_seed': 42}
logger.info(f"Using parameters: {params}")

# Logger error example
try:
    raise Exception("This is a test exception for logging purposes.")
except Exception as e:
    logger.error("An error occurred", exc_info=True)

# Logger warning example
accuracy = 0.8
logging.warning(f"Model accuracy ({accuracy:.2f}) is below acceptable threshold!")

