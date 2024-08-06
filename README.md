### 환경 세팅

**1. anaconda 환경 세팅**

설치는 [링크](https://docs.anaconda.com/anaconda/install/)를 참고해주세요.

```bash
conda env create -f environment.yml
conda activate stock-value-crawler
```

**2. Chrome Driver 설치**

[링크](https://developer.chrome.com/docs/chromedriver/downloads?hl=ko)를 참고하여 크롬 드라이버를 설치 후 bin 파일의 경로를 파악합니다.


### 코드 실행 방법

```bash
python3 main.py --dt 20240806 --top_k 5 --file_path stocks.csv
```

다음과 같은 3가지 argument로 구성됩니다.
1. `--dt`
   * dt에 해당하는 날짜의 종목코드를 가져옵니다. YYYYMMDD 형식으로 입력합니다.
   * 기본값은 오늘 날짜로 설정되며, 생략 가능합니다.
2. `--top_k`
   * 시장(코스피/코스닥)별로 몇 개의 상위 종목을 가져올지를 의미하는 파라미터입니다.
   * 기본값은 5로 설정되어 있습니다.
3. `--file_path`
   * 결과 파일을 저장할 경로를 의미합니다.
   * 설정하지 않을 경우 `{dt}_stocks.csv` 형식으로 저장됩니다. (ex) 20240806_stocks.csv)