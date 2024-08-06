#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########
# imports #
###########
import re
import logging
from typing import List, Any
from tqdm.asyncio import tqdm


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt=time_format)

    # Create a logger and set the custom formatter
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set the log level (optional, can be DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(level)
    return logger


async def run_in_batches(tasks: List[Any], batch_size: int, logger: logging.Logger = get_logger(__name__)) -> List[Any]:
    results = []
    num_batch = (len(tasks) // batch_size) if (len(tasks) % batch_size == 0) else (len(tasks) // batch_size + 1)
    for i in range(0, num_batch):
        start = i * batch_size
        end = min(len(tasks), (i+1) * batch_size)
        batch = tasks[start:end]
        results.extend(await tqdm.gather(
            *batch,
            desc=f"Processing batch {i+1}/{num_batch}({100*(i+1)/num_batch:.1f}%)"
        ))
        logger.info(f"Finished batch {i+1}/{num_batch}({100*(i+1)/num_batch:.1f}%)")
    return results


def extract_column_name(s):
    return re.match(r'^[^ ]+', s).group(0)