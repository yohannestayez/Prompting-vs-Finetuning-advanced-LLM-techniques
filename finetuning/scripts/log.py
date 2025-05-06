import logging

def get_file_logger(log_file="finetuning/training.log"):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Add encoding="utf-8" to the FileHandler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
