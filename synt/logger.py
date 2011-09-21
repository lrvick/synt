# -*- coding: utf-8 -*-
import logging

def create_logger(name, level=logging.DEBUG):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #console handling
    ch = logging.StreamHandler()
    ch.setLevel(level)

    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(log_formatter)

    logger.addHandler(ch)

    return logger
