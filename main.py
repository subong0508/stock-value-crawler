#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########
# imports #
###########
import io
import os
import sys
import random
import asyncio
from dataclasses import dataclass

from tqdm.asyncio import tqdm
from typing import List, Optional, Tuple
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pykrx import stock
from pandas import DataFrame
from bs4 import BeautifulSoup

from util import get_logger, fetch_page_content, extract_column_name, DIV_IDS

#############
# constants #
#############
MARKET_LIST = ["KOSPI", "KOSDAQ"]
STOCK_REPORT_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={stock_code}"

COLUMNS = [
    "재무연월", "market", "종목코드", "종목명", "매출액", "YoY", "영업이익", "당기순이익",
    "EPS", "BPS", "PER", "PBR", "ROE", "EV/EBITDA"
]

#############
# variables #
#############
logger = get_logger(__name__)
concurrency_cnt = os.cpu_count() * 2


@dataclass
class Stock:
    market: str
    stock_code: str
    stock_name: str


def get_stocks(dt: str) -> List[Stock]:
    stocks = []
    for market in MARKET_LIST:
        stock_codes = stock.get_market_ticker_list(dt, market=market)
        stocks += [
            Stock(market=market, stock_code=stock_code, stock_name=stock.get_market_ticker_name(stock_code))
            for stock_code in stock_codes
        ]
    return stocks


async def scrape(stock_code: str) -> Optional[List[BeautifulSoup]]:
    url = STOCK_REPORT_URL.format(stock_code=stock_code)
    content = await fetch_page_content(url, logger=logger)
    if content is not None:
        tags = [
            (
                BeautifulSoup(content, "html.parser")
                .find("div", id=DIV_IDS[0])
                .find("div", {"class": "wrapper-table"})
                .find("div", {"class": "cmp-table-div"})
                .find("table")
            ),
            BeautifulSoup(content, "html.parser").find("div", id=DIV_IDS[1]).find("table")
        ]
        return tags


async def bounded_scrape(stock_code: str, semaphore: asyncio.Semaphore, progress_bar):
    async with semaphore:
        result = await scrape(stock_code)
        progress_bar.update(1)
        return result


async def get_stock_values(stocks: List[Stock], debug: bool = False) -> DataFrame:
    if debug:
        random.shuffle(stocks)
        stocks = stocks[:100]

    stock_codes = [s.stock_code for s in stocks]
    semaphore = asyncio.Semaphore(concurrency_cnt)
    logger.info(f"웹 스크래핑 시작.. 동시실행 수 : {concurrency_cnt}")
    progress_bar = tqdm(total=len(stock_codes), desc="Scraping progress", unit="stock")
    tasks = [bounded_scrape(stock_code, semaphore, progress_bar) for stock_code in stock_codes]
    table_results = await asyncio.gather(*tasks)

    pattern = r"\d{4}\(E\)"
    stock_values = pd.DataFrame()
    for stock_, tables in zip(stocks, table_results):
        market, stock_code, stock_name = stock_.market, stock_.stock_code, stock_.stock_name
        if tables:
            table1, table2 = tables

            items = pd.read_html(io.StringIO(str(table1)))[0].iloc[1].values[0].split()
            today_df = pd.DataFrame({
                k: [float(items[i+1].replace(",", ""))]
                if items[i+1].replace(",", "").lstrip("-").isdigit() else [np.nan]
                for i, k in enumerate(items) if k in COLUMNS
            })
            today_df = today_df.assign(재무연월="TODAY")
            today_df = today_df.reindex(columns=COLUMNS)

            future_df = pd.read_html(io.StringIO(str(table2)))[0]
            future_df.columns = [extract_column_name(col[1]) for col in future_df.columns]
            future_df = future_df[future_df["재무년월"].str.contains(pattern, regex=True)]
            if not future_df.empty:
                future_df = future_df.rename(columns={"금액": "매출액", "재무년월": "재무연월"})
                future_df = future_df.reindex(columns=COLUMNS)
                df = pd.concat([today_df, future_df])
                df = df.assign(market=market)
                df = df.assign(종목코드=stock_code)
                df = df.assign(종목명=stock_name)
                stock_values = pd.concat([stock_values, df])
    return stock_values


def extract_top_stocks(stock_values: DataFrame, algorithm: str, top_k: int) -> List[str]:
    top_groups = []

    def _rank(market: str):
        nonlocal top_groups

        col = algorithm
        stock_values_ = stock_values[stock_values["market"] == market]
        if col == "YoY":
            future_values = stock_values_[stock_values_["재무연월"] != "TODAY"]
            grouped_sum = future_values.groupby(["종목코드"])["YoY"].sum().reset_index()
            grouped_sum[f"YoY_rank"] = grouped_sum[col].rank(method="min", ascending=False)
            top_groups += grouped_sum.sort_values(f"YoY_rank").head(top_k)["종목코드"].unique().tolist()
        elif col == "EPS":
            next_year = f"{datetime.today().year + 1}(E)"
            today_values = stock_values_[stock_values_["재무연월"] == "TODAY"]
            future_values = stock_values_[stock_values_["재무연월"] == next_year]
            total_values = pd.merge(
                today_values[["종목코드", "EPS"]],
                future_values[["종목코드", "EPS"]],
                on="종목코드",
                suffixes=("_today", "_future")
            )
            total_values["EPS_ratio"] = total_values["EPS_future"] / total_values["EPS_today"]
            total_values["EPS_ratio"] = total_values["EPS_ratio"].replace([-np.inf, np.inf, np.nan], 0)
            top_groups += (
                total_values.sort_values("EPS_ratio", ascending=False).head(top_k)["종목코드"].unique().tolist()
            )

    for market in MARKET_LIST:
        _rank(market)
    return top_groups


def save_results(stock_values: DataFrame, valuable_stocks: List[str], file_path: str):
    df = stock_values.copy()
    df = df[df["종목코드"].isin(valuable_stocks)]
    df["rank1"] = pd.Categorical(df["종목코드"], categories=valuable_stocks, ordered=True)
    df["rank2"] = pd.Categorical(df["재무연월"], categories=["TODAY"]+sorted(df["재무연월"].unique())[:-1], ordered=True)
    df = df.sort_values(["market", "rank1", "rank2"], ascending=[False, True, True])
    df["종목코드"] = df["종목코드"].apply(lambda x: f'="{x}"')
    df = df[COLUMNS]
    encoding = "cp949" if sys.platform.startswith("win") else "utf-8"
    df.to_csv(file_path, index=False, encoding=encoding, quoting=0)


async def main():
    parser = ArgumentParser(description="Stock value crawler for valuable stocks in South Korea.")
    parser.add_argument(
    "--dt", help="date time. ex) '20240731'", metavar="YYYYMMDD",
    )
    parser.add_argument(
        "--top_k", help="number of stocks to be ranked", type=int, default=5,
    )
    parser.add_argument(
        "--algorithm", help="algorithm to rank stocks", choices=["YoY", "EPS"], type=str, required=True
    )
    parser.add_argument("--file_path", help="file path where result is stored", type=str)
    parser.add_argument("--debug", help="enable debug", action="store_true")
    args = parser.parse_args()

    dt = args.dt
    if dt is None:
        dt = datetime.now().strftime("%Y%m%d")

    file_path = args.file_path
    if file_path is None:
        file_path = f"{dt}_top_stocks_by_{args.algorithm}.csv"

    stocks = get_stocks(dt)
    logger.info(f"오늘자 주식 종목 가져오기 완료. 종목 수 : {len(stocks)}")

    stock_values = await get_stock_values(stocks, args.debug)
    logger.info(f"미래 주식 가치 스크래핑 완료. 종목 수 : {len(stock_values['종목코드'].unique())}")

    valuable_stocks = extract_top_stocks(stock_values, args.algorithm, args.top_k)
    if not args.debug:
        save_results(stock_values, valuable_stocks, file_path)
        logger.info(f"결과 파일 {file_path}에 저장 완료.")


if __name__ == "__main__":
    asyncio.run(main())