#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########
# imports #
###########
import io
import os
import asyncio
from tqdm.asyncio import tqdm
from typing import List, Optional
from datetime import datetime
from argparse import ArgumentParser

import pandas as pd
from pykrx import stock
from pandas import DataFrame
from bs4 import BeautifulSoup

from util import get_logger, fetch_page_content, extract_column_name, DIV_ID

#############
# constants #
#############
MARKET_LIST = ["KOSPI", "KOSDAQ"]
STOCK_REPORT_URL = "https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={stock_code}"

TODAY_VALUES_COLUMNS = ["재무년월일", "market", "종목코드", "종목명", "EPS", "BPS", "PER", "PBR"]
FUTURE_VALUES_COLUMNS = [
    "재무년월일", "market", "종목코드", "종목명", "매출액", "YoY", "영업이익", "당기순이익",
    "EPS", "BPS", "PER", "PBR", "ROE", "EV/EBITDA"
]

#############
# variables #
#############
logger = get_logger(__name__)
concurrency_cnt = os.cpu_count() * 5


def get_today_stock_values(dt: str) -> DataFrame:
    df = pd.DataFrame()
    for market in MARKET_LIST:
        df_temp = stock.get_market_fundamental(dt, market=market)
        df_temp = df_temp.assign(market=market)
        df = pd.concat([df, df_temp])
    df = df.assign(재무년월일="TODAY")
    df = df.reset_index().rename(columns={"티커": "종목코드"})
    df["종목명"] = df["종목코드"].apply(stock.get_market_ticker_name)
    return df[TODAY_VALUES_COLUMNS]


async def scrape(stock_code: str) -> Optional[BeautifulSoup]:
    url = STOCK_REPORT_URL.format(stock_code=stock_code)
    content = await fetch_page_content(url, logger=logger)
    if content is not None:
        tag = BeautifulSoup(content, "html.parser").find("div", id=DIV_ID).find("table")
        return tag


async def bounded_scrape(stock_code: str, semaphore: asyncio.Semaphore, progress_bar):
    async with semaphore:
        result = await scrape(stock_code)
        progress_bar.update(1)
        return result


async def get_future_stock_values(today_values: DataFrame):
    stocks = today_values[["market", "종목코드", "종목명"]].to_dict(orient="records")
    stock_codes = [s["종목코드"] for s in stocks]

    semaphore = asyncio.Semaphore(concurrency_cnt)
    logger.info(f"웹 스크래핑 시작.. 동시실행 수 : {concurrency_cnt}")
    progress_bar = tqdm(total=len(stock_codes), desc="Scraping progress", unit="stock")
    tasks = [bounded_scrape(stock_code, semaphore, progress_bar) for stock_code in stock_codes]
    table_results = await asyncio.gather(*tasks)

    pattern = r"\d{4}\(E\)"
    future_values = pd.DataFrame()
    for stock, table in zip(stocks, table_results):
        market, stock_code, stock_name = stock["market"], stock["종목코드"], stock["종목명"]
        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]
            df.columns = [extract_column_name(col[1]) for col in df.columns]
            df = df[df["재무년월"].str.contains(pattern, regex=True)]
            if not df.empty:
                df = df.assign(market=market)
                df = df.assign(종목코드=stock_code)
                df = df.assign(종목명=stock_name)
                future_values = pd.concat([future_values, df])

    future_values = future_values.rename(columns={"금액": "매출액", "재무년월": "재무년월일"})
    future_values = future_values.reindex(columns=FUTURE_VALUES_COLUMNS)
    return future_values


def extract_valuable_stocks(future_values: DataFrame, col: str, top_k: int) -> List[str]:
    top_groups = []
    for market in MARKET_LIST:
        grouped_sum = future_values[future_values["market"] == market].groupby(["종목코드"])[col].sum().reset_index()
        grouped_sum[f"{col}_rank"] = grouped_sum[col].rank(method="min", ascending=False)
        grouped_sum = grouped_sum.sort_values(f"{col}_rank")
        top_groups += grouped_sum[grouped_sum[f"{col}_rank"] <= top_k]["종목코드"].unique().tolist()
    return top_groups


def save_results(today_values: DataFrame, future_values: DataFrame, valuable_stocks: List[str], file_path: str):
    today_values = today_values.reindex(columns=FUTURE_VALUES_COLUMNS)
    df = pd.concat([today_values, future_values], ignore_index=True)
    df = df[df["종목코드"].isin(valuable_stocks)]
    df["rank1"] = pd.Categorical(df["종목코드"], categories=valuable_stocks, ordered=True)
    df["rank2"] = pd.Categorical(df["재무년월일"], categories=["TODAY"]+sorted(df["재무년월일"].unique())[:-1], ordered=True)
    df = df.sort_values(["rank1", "rank2"])
    df = df[FUTURE_VALUES_COLUMNS]
    df.to_csv(file_path, index=False)


async def main():
    parser = ArgumentParser(description="Stock value crawler for valuable stocks in South Korea.")
    parser.add_argument(
    "--dt", help="date time. ex) '20240731'", metavar="YYYYMMDD",
    )
    parser.add_argument(
        "--top_k", help="number of stocks to be ranked", type=int, default=5,
    )
    parser.add_argument("--file_path", help="file path where result is stored", type=str)
    args = parser.parse_args()

    dt = args.dt
    if dt is None:
        dt = datetime.now().strftime("%Y%m%d")
    top_k = args.top_k
    file_path = args.file_path
    if file_path is None:
        file_path = f"{dt}_stocks.csv"

    today_values = get_today_stock_values(dt)
    logger.info(f"오늘자 주식 종목 가져오기 완료. 종목 수 : {today_values.shape[0]}")

    future_values = await get_future_stock_values(today_values)
    logger.info(f"미래 주식 가치 스크래핑 완료. 종목 수 : {len(future_values['종목코드'].unique())}")

    valuable_stocks = extract_valuable_stocks(future_values, "YoY", top_k)
    save_results(today_values, future_values, valuable_stocks, file_path)
    logger.info(f"결과 파일 {file_path}에 저장 완료.")


if __name__ == "__main__":
    asyncio.run(main())