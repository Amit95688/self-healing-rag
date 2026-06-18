import logging
import os
from datetime import datetime


LOG_FILE = f"app_{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)

logging.info("Logging setup complete.")

if __name__ == "__main__":
    logging.info("Logging setup complete.")