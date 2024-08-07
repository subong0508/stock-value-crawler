### 환경 세팅

anaconda 설치는 [링크](https://docs.anaconda.com/anaconda/install/)를 참고해주세요.

```bash
conda create -n stock-value-crawler python=3.10
conda activate stock-value-cralwer
pip install -r requirements.txt
playwright install
```

### 코드 실행 방법

```bash
python3 main.py --dt 20240806 --top_k 5 --algorithm EPS --file_path stocks.csv
```

또는

```bash
python main.py --dt 20240806 --top_k 5 --algorithm EPS --file_path stocks.csv
```

다음과 같은 3가지 argument로 구성됩니다.
1. `--dt`
   * dt에 해당하는 날짜의 종목코드를 가져옵니다. YYYYMMDD 형식으로 입력합니다.
   * 기본값은 오늘 날짜로 설정되며, 생략 가능합니다.
2. `--top_k`
   * 시장(코스피/코스닥)별로 몇 개의 상위 종목을 가져올지를 의미하는 파라미터입니다.
   * 기본값은 5로 설정되어 있습니다.
3. `--algorithm`
   * 필수 입력값으로, EPS 또는 YoY 중 하나를 입력합니다.
4. `--file_path`
   * 결과 파일을 저장할 경로를 의미합니다.
   * 설정하지 않을 경우 `{dt}_top_stocks_by_{algorithm}.csv` 형식으로 저장됩니다. (ex) 20240806_top_stocks_by_EPS.csv)
5. `--debug`
   * 디버깅용 변수이며, `--debug` 인자를 입력하면 100개 종목에 대해서만 컨센서스 테이블을 스크래핑하며 결과물을 파일로 따로 저장하지 않습니다.