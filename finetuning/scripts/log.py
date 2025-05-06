import logging

def get_file_logger(log_file="training.log"):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if get_file_logger is called multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
