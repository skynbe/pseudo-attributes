import os
import os.path
import hashlib
import errno
import logging
from collections import defaultdict
from string import Formatter
import torch
import re, pdb
from datetime import datetime

import cv2
import numpy as np
import pdb


def get_logger(name, fmt='%(asctime)s:%(name)s:%(message)s',
               print_level=logging.INFO,
               write_level=logging.DEBUG, log_file='', mode='w'):
    """
    Get Logger with given name
    :param name: logger name.
    :param fmt: log format. (default: %(asctime)s:%(levelname)s:%(name)s:%(message)s)
    :param level: logging level. (default: logging.DEBUG)
    :param log_file: path of log file. (default: None)
    :return:
    """
    logger = logging.getLogger(name)
    #  logger.setLevel(write_level)
    logging.basicConfig(level=print_level)
    formatter = logging.Formatter(fmt, datefmt='%Y/%m/%d %H:%M:%S')

    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(write_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    if print_level is not None:
        try:
            import coloredlogs
            coloredlogs.install(level=print_level, logger=logger)
            coloredlogs.DEFAULT_LEVEL_STYLES = {'critical': {'color': 'red', 'bold': True}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'color': 'green', 'bold': True}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
        except ImportError:
            print("Please install Coloredlogs for better view")
            # Add stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(print_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    return logger