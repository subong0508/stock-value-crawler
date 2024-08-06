#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########
# imports #
###########
import re
import asyncio
import logging
from typing import Optional
from playwright.async_api import async_playwright, TimeoutError

#############
# constants #
#############
DIV_ID = "frmFS1"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    time_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt=time_format)

    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level)
    return logger


async def fetch_page_content(
        url: str,
        logger: logging.Logger,
        retries: int = 3
) -> Optional[str]:
    async with async_playwright() as p:
        for attempt in range(1, retries + 1):
            browser = None
            try:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.set_extra_http_headers(HEADERS)
                try:
                    await page.goto(url, timeout=60000)
                    await page.wait_for_selector(f"div#{DIV_ID}", timeout=10000)
                    content = await page.content()
                    return content
                except TimeoutError as e:
                    logger.error(f"Timeout error on attempt {attempt} for URL {url}: {e}")
                except Exception as e:
                    logger.error(f"Exception on attempt {attempt} for URL {url}: {e}")
                finally:
                    if page:
                        await page.close()
            except Exception as e:
                logger.error(f"Exception on attempt {attempt} when launching browser for URL {url}: {e}")
            finally:
                if browser:
                    await browser.close()
            await asyncio.sleep(2)

        logger.error(f"Failed to fetch page content for {url} after {retries} attempts.")
        return None



def extract_column_name(s):
    return re.match(r'^[^ ]+', s).group(0)