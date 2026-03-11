# Importing logging module from the logger defined earlier
from Logger import logging

# Use the Logger in Your Script
logging.info("Script started.")

# Logger information
params = {'test_size': 0.2, 'random_seed': 42}
logging.info(f"Using parameters: {params}")

# Logger error example
try:
    raise Exception("This is a test exception for logging purposes.")
except Exception as e:
    logging.error("An error occurred", exc_info=True)

# Logger warning example
accuracy = 0.8
logging.warning(f"Model accuracy ({accuracy:.2f}) is below acceptable threshold!")

# Critical error example
logging.critical("This is a critical error.")