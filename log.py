import logging


def setup_log(log_ident,log_file, log_level=logging.INFO):
    logger = logging.getLogger(log_ident)
    logger.setLevel(logging.INFO)
    
    # log file
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(handler)

    
    return logger