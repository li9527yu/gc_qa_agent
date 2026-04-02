import logging
import os
from datetime import datetime

def setup_logger():
    log_dir = "logs/api_server"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"api_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("easy_rag_api")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
