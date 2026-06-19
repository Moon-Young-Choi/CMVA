## 0. 전체 방향 재정의

첨부 논의의 기존 CMVA는 암호화폐 시장데이터 수집, GARCH 계열 변동성 모델, rolling correlation/PCA 기반 시장국면 분류, regime-aware volatility targeting 백테스트, CLI/리포트 자동화가 핵심이었습니다. 

이제 요구사항을 반영하면 프로젝트는 이렇게 바뀌어야 합니다.

> **CMVA는 여러 개의 CLI 명령어를 순서대로 실행하는 도구가 아니라, `cmva` 하나로 실행되는 터미널 기반 실시간 리서치 앱이다. 앱을 켜두면 공개 암호화폐 데이터를 계속 수집하고, closed candle 단위로 데이터 검증·feature 갱신·GARCH 변동성 forecast·시장국면 판단·paper backtest를 반복 수행하며, 사용자는 화살표와 Enter로 화면과 작업을 통제한다.**

즉, 개발 목표는 **command-line script 모음**이 아니라 **interactive TUI app**입니다. 여기서 TUI는 terminal user interface입니다. Python에서는 Textual이 이런 목적에 잘 맞습니다. Textual은 Python으로 터미널 또는 브라우저에서 실행 가능한 TUI를 만들 수 있는 프레임워크이고, 키 입력 이벤트도 처리할 수 있습니다. ([Textual Documentation][1])

---

# 1. Rolling Window란?

**Rolling window**는 “최근 N개의 데이터만 보면서 계속 움직이는 분석 구간”입니다.

예를 들어 CMVA가 1시간 candle 데이터를 쓰고, rolling window를 `720`으로 잡는다면:

```text
720개 hourly candle = 720시간 = 30일
```

입니다.

2026-06-19 13:00 시점에 rolling correlation을 계산한다면, 대략 직전 720시간 데이터만 사용합니다. 2026-06-19 14:00 candle이 새로 닫히면 가장 오래된 1시간 데이터는 빠지고, 새 1시간 데이터가 들어옵니다.

```text
이전 window:
[t-719, ..., t-1, t]

새 window:
[t-718, ..., t, t+1]
```

이 방식의 목적은 전체 과거를 한꺼번에 고정해서 보는 게 아니라, **최근 시장 상태가 어떻게 바뀌고 있는지 추적**하는 것입니다. statsmodels의 RollingOLS도 고정된 관측치 window에 OLS를 적용한 뒤 그 window를 이동시키는 방식으로 설명됩니다. ([Statsmodels][2])

CMVA에서 rolling window가 쓰이는 곳은 다음입니다.

| 항목                  |       예시 window | 의미                          |
| ------------------- | --------------: | --------------------------- |
| Rolling volatility  |    24h, 7d, 30d | 최근 변동성                      |
| Rolling correlation | 72h, 168h, 720h | 최근 자산 간 동조화                 |
| Rolling beta        |      168h, 720h | BTC 또는 시장 basket에 대한 민감도    |
| PCA rolling window  |            720h | 시장 공통요인 지배력                 |
| Regime threshold    |         720h 이상 | 현재 변동성·상관이 과거 대비 높은지 낮은지 판단 |

중요한 규칙은 하나입니다.

> **t 시점의 rolling feature는 반드시 t 시점까지의 데이터만 사용해야 한다.**

`t+1`의 수익률이나 candle을 feature 계산에 넣으면 look-ahead bias가 생깁니다.

---

# 2. Candle이란?

**Candle**, 또는 **candlestick/kline**, 은 특정 시간 구간의 가격 움직임을 요약한 bar입니다.

Binance Spot API 문서에서 kline/candlestick data는 symbol별 bar이고, 각 kline은 open time으로 식별됩니다. 응답 필드에는 open time, open price, high price, low price, close price, volume, close time 등이 들어갑니다. 또한 `1h` interval도 지원됩니다. ([Binance Developers][3])

예를 들어 `BTCUSDT`, `1h` candle은 다음 정보를 가집니다.

| 필드         | 의미                |
| ---------- | ----------------- |
| open time  | candle 시작 시각      |
| open       | 해당 1시간의 첫 거래 가격   |
| high       | 해당 1시간의 최고 가격     |
| low        | 해당 1시간의 최저 가격     |
| close      | 해당 1시간의 마지막 거래 가격 |
| volume     | 해당 1시간의 거래량       |
| close time | candle 종료 시각      |

예를 들어 13:00 candle은 대략 이런 뜻입니다.

```text
13:00:00 ~ 13:59:59 동안의 가격 요약

open  = 13:00 직후 첫 가격
high  = 그 1시간 동안의 최고 가격
low   = 그 1시간 동안의 최저 가격
close = 13:59:59 근처 마지막 가격
volume = 그 1시간 동안 거래된 수량
```

CMVA에서는 **closed candle만 모델 입력으로 사용**해야 합니다. WebSocket으로는 현재 진행 중인 candle도 계속 업데이트되지만, 그 candle은 아직 확정되지 않았습니다. Binance Spot WebSocket의 kline stream은 현재 candle 업데이트를 push하고, payload 안에 해당 kline이 닫혔는지를 나타내는 `x` 필드가 있습니다. ([Binance Developers][4])

따라서 CMVA의 원칙은 이겁니다.

```text
x = false  → 화면 표시용 current candle
x = true   → 저장, 검증, feature 계산, forecast, regime 판단, backtest 업데이트에 사용
```

---

# 3. forecast가 이 경우에는 무엇을 의미하는가?

CMVA에서 **forecast**는 “가격이 오른다/내린다”는 방향성 예측이 아닙니다.

여기서 forecast는 주로 다음을 뜻합니다.

> **다음 1시간 동안의 조건부 변동성 또는 조건부 분산 예측**

수식으로 쓰면 다음입니다.

```text
forecast variance:
σ²_{t+1|t}

forecast volatility:
σ_{t+1|t}
```

즉, `t` 시점까지의 데이터로 `t+1` 시점의 위험 수준을 추정하는 것입니다.

GARCH(1,1)의 기본 구조는 다음과 같습니다.

```text
r_t = μ + ε_t
ε_t = σ_t z_t
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```

`arch` 문서도 ARCH/GARCH류 모델을 평균모형, 변동성과정, 표준화 잔차분포로 나누며, 관측 수익률 또는 잔차를 volatility shock으로 사용한다고 설명합니다. ([Arch Docs][5])

CMVA에서 forecast가 이런 값으로 나온다고 해봅시다.

```text
BTCUSDT forecast_vol_1h = 0.012
```

이건 다음 뜻입니다.

```text
다음 1시간 BTCUSDT 수익률의 표준편차가 약 1.2%로 예상된다.
```

이 말은 아닙니다.

```text
BTC가 다음 1시간에 1.2% 오른다.
```

CMVA는 가격 방향성 모델이 아니라 **위험 예측 모델**입니다. 따라서 forecast는 포지션 크기를 조절하는 데 사용됩니다.

```text
forecast volatility 높음 → 포지션 축소
forecast volatility 낮음 → 포지션 정상화 또는 확대
```

예를 들어:

```text
base_weight_t = target_vol_per_period / forecast_vol_t
```

입니다. 목표 변동성이 고정되어 있다면, forecast volatility가 높아질수록 weight가 작아집니다.

CMVA에서 forecast는 세 층으로 나눌 수 있습니다.

| forecast 종류                     | 의미                              | 사용처                  |
| ------------------------------- | ------------------------------- | -------------------- |
| Asset forecast vol              | 각 코인의 다음 1시간 변동성                | 종목별 위험 측정            |
| Basket forecast vol             | equal-weight basket의 다음 1시간 변동성 | 포트폴리오 총노출 결정         |
| Regime forecast 또는 regime label | 현재 시장이 systemic risk인지 판단       | regime multiplier 적용 |

MVP에서는 **GARCH 기반 1h volatility forecast**가 핵심이고, 나중에 HAR, EWMA, ML 모델 등을 같은 인터페이스로 추가하면 됩니다.

---

# 4. 변동성 shock인지 아닌지는 어떻게 판단하나?

여기서는 “변동성 stock”을 **volatility shock**으로 해석하겠습니다.

먼저 구분해야 합니다.

| 개념                     | 의미                   |
| ---------------------- | -------------------- |
| High volatility regime | 변동성이 한동안 높은 상태       |
| Volatility shock       | 방금 발생한 비정상적으로 큰 충격   |
| Systemic shock         | 여러 코인이 동시에 충격을 받은 상태 |
| Idiosyncratic shock    | 특정 코인만 크게 흔들린 상태     |

CMVA에서는 shock 판단을 단일 기준 하나로 하지 말고, **모델 기준 + 경험적 분위수 기준 + 횡단면 확인**을 같이 쓰는 게 좋습니다.

## 4.1 1단계: standardized shock score

`t-1` 시점에 예측한 변동성으로 `t` 시점의 실제 수익률을 표준화합니다.

```text
shock_score_t = |r_t - μ_{t|t-1}| / σ_{t|t-1}
```

간단히 평균수익률 `μ`를 0으로 보면:

```text
shock_score_t = |r_t| / forecast_vol_{t|t-1}
```

예를 들어:

```text
forecast_vol = 1.0%
actual return = -4.2%

shock_score = 4.2
```

이면 매우 큰 shock입니다.

기본 규칙은 이렇게 둘 수 있습니다.

| shock_score | 해석             |
| ----------: | -------------- |
|     `< 2.0` | 정상 범위          |
| `2.0 ~ 3.0` | moderate shock |
|     `> 3.0` | severe shock   |

다만 암호화폐 수익률은 fat-tail이 강할 수 있으므로, 고정된 `3σ` 규칙만 믿기보다는 rolling empirical quantile도 같이 쓰는 게 좋습니다.

## 4.2 2단계: realized volatility jump

단일 candle 수익률만 보는 것이 아니라, 최근 몇 시간의 realized volatility가 평소보다 얼마나 튀었는지도 봅니다.

```text
rv_6h_t = std(r_{t-5}, ..., r_t)
rv_jump_ratio_t = rv_6h_t / median(rv_6h over trailing 30d)
```

예시 기준:

| 조건                                | 판정                     |
| --------------------------------- | ---------------------- |
| `rv_jump_ratio > 2.0`             | volatility jump        |
| `rv_jump_ratio > 3.0`             | severe volatility jump |
| `rv_6h`가 trailing 30d의 95% 분위수 초과 | high-vol shock         |

## 4.3 3단계: cross-sectional shock 확인

개별 코인 하나만 튄 것인지, 시장 전체가 같이 흔들린 것인지 구분해야 합니다.

```text
shock_breadth_t =
shock가 발생한 코인 수 / 전체 코인 수
```

예를 들어 universe가 10개 코인이고, 그중 7개가 동시에 shock이면:

```text
shock_breadth = 70%
```

이 경우는 단순 개별 이슈가 아니라 systemic shock일 가능성이 큽니다.

## 4.4 최종 shock classification

CMVA에서는 다음처럼 판단하면 됩니다.

```text
if shock_score high
   and shock_breadth high
   and average_corr high:
       shock_type = SYSTEMIC_VOL_SHOCK

elif shock_score high
   and shock_breadth low
   and dispersion high:
       shock_type = IDIOSYNCRATIC_VOL_SHOCK

elif realized_vol_jump high
   but shock_score not extreme:
       shock_type = VOL_REGIME_BUILDUP

else:
       shock_type = NORMAL
```

정리하면:

| 상태                        | 조건                                | 해석            |
| ------------------------- | --------------------------------- | ------------- |
| `NORMAL`                  | shock score 낮음                    | 정상            |
| `MODERATE_SHOCK`          | 일부 자산의 standardized shock 상승      | 주의            |
| `IDIOSYNCRATIC_VOL_SHOCK` | 특정 코인만 급변, dispersion 높음          | 개별 이슈         |
| `SYSTEMIC_VOL_SHOCK`      | 다수 코인 동시 급변, correlation 높음       | 시장 전체 위험      |
| `VOL_REGIME_BUILDUP`      | 단발 shock은 아니지만 realized vol 상승 지속 | 고변동성 국면 진입 가능 |

이 판단은 **현재 진행 중인 candle**이 아니라 **닫힌 candle** 기준으로 해야 합니다.

---

# 5. “켜놓으면 실시간으로 수집·분석·예측·백테스팅”은 어떻게 설계하나?

가능합니다. 다만 정확히 표현하면 다음과 같습니다.

> CMVA는 실시간으로 시장데이터를 수집하고, closed 1h candle이 생길 때마다 feature·forecast·regime·paper backtest를 갱신한다. 단, 백테스트는 미래를 볼 수 없으므로 “실시간 백테스트”가 아니라 **walk-forward paper backtest의 incremental update**로 구현한다.

즉, 앱을 켜두면 다음 사이클이 반복됩니다.

```text
1. WebSocket으로 현재 candle 업데이트 수신
2. candle이 아직 안 닫혔으면 화면만 갱신
3. candle이 닫히면 저장
4. 데이터 검증
5. rolling feature 갱신
6. GARCH forecast 갱신
7. shock/regime 판단
8. 다음 1시간 목표 exposure 계산
9. 이전 1시간의 paper position 성과를 확정
10. dashboard와 backtest 결과 갱신
```

Binance Spot REST API의 `/api/v3/klines`는 historical 또는 recent kline 데이터를 가져오는 데 적합하고, Spot WebSocket kline stream은 현재 kline/candlestick 업데이트를 push합니다. 따라서 CMVA는 시작 시 REST로 과거 데이터를 보충하고, 실행 중에는 WebSocket으로 최신 candle을 받는 구조가 자연스럽습니다. ([Binance Developers][3])

---

# 6. 새 CMVA 개발계획

## 6.1 최종 제품 형태

기존처럼 이렇게 쓰지 않습니다.

```bash
python -m cmva collect
python -m cmva validate-data
python -m cmva features
python -m cmva garch-forecast
python -m cmva regime
python -m cmva backtest
```

최종 사용자는 이렇게 실행합니다.

```bash
cmva
```

또는:

```bash
python -m cmva
```

실행하면 interactive terminal app이 열립니다.

```text
CMVA - Crypto Market Volatility Analysis

[Dashboard] [Data] [Features] [Models] [Regime] [Backtest] [Settings] [Logs]

Latest closed candle: 2026-06-19 14:00 UTC
Universe: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT
Current regime: SYSTEMIC_RISK
Shock status: SYSTEMIC_VOL_SHOCK
Basket forecast vol 1h: 1.43%
Target exposure next hour: 0.27x
Backtest net cumulative return: 38.4%
Backtest max drawdown: -18.6%
```

사용자는 화살표와 Enter로 이동합니다.

| 키         | 기능                      |
| --------- | ----------------------- |
| `←` / `→` | 탭 이동                    |
| `↑` / `↓` | 항목 이동                   |
| `Enter`   | 선택/실행                   |
| `Space`   | 일시정지/재개                 |
| `R`       | 현재 데이터 기준 즉시 refresh    |
| `F`       | GARCH 강제 refit          |
| `B`       | historical backtest 재실행 |
| `E`       | report export           |
| `Q`       | 종료                      |

---

## 6.2 내부적으로는 command pipeline을 유지하되, 사용자에게 숨긴다

겉으로는 하나의 앱이지만 내부는 여전히 pipeline입니다.

```text
CMVA App
 ├─ DataService
 ├─ ValidationService
 ├─ FeatureService
 ├─ ModelService
 ├─ ForecastService
 ├─ ShockService
 ├─ RegimeService
 ├─ PolicyService
 ├─ BacktestService
 ├─ ReportService
 └─ TuiController
```

사용자는 여러 명령어를 외우지 않습니다.

대신 앱 내부에서 다음 작업이 자동으로 돌아갑니다.

```text
startup:
  load config
  load local cache
  fetch missing historical candles
  validate historical data
  compute initial features
  fit or load GARCH model
  run initial historical backtest
  start WebSocket stream
  open TUI dashboard

on every websocket kline update:
  update current candle panel

on closed candle:
  persist candle
  validate new candle
  update rolling features
  update forecast
  classify shock
  classify regime
  compute next target exposure
  settle previous paper-backtest step
  refresh dashboard

periodic:
  refit GARCH every 24 closed candles
  rerun model selection weekly or manually
  export report on demand
```

---

# 7. 새 아키텍처

## 7.1 패키지 구조

```text
cmva/
  __init__.py
  __main__.py

  app.py
  config.py
  state.py
  events.py
  logging_config.py

  tui/
    __init__.py
    app.py
    screens.py
    widgets.py
    bindings.py
    theme.py

  data/
    __init__.py
    rest_client.py
    websocket_client.py
    candle.py
    storage.py
    validation.py
    universe.py

  features/
    __init__.py
    returns.py
    volatility.py
    correlation.py
    pca.py
    dispersion.py
    rolling_beta.py

  models/
    __init__.py
    base.py
    garch.py
    registry.py
    diagnostics.py
    selection.py

  forecast/
    __init__.py
    volatility_forecaster.py
    portfolio_vol.py

  regime/
    __init__.py
    shock.py
    classifier.py
    thresholds.py

  policy/
    __init__.py
    base.py
    vol_target.py
    regime_vol_target.py

  backtest/
    __init__.py
    engine.py
    live_paper.py
    benchmarks.py
    costs.py
    metrics.py

  reports/
    __init__.py
    markdown.py
    html.py
    plots.py

  tests/
    test_candle_validation.py
    test_no_lookahead.py
    test_shock_classifier.py
    test_regime_classifier.py
    test_backtest_engine.py
    test_policy_shift.py
```

---

## 7.2 핵심 객체

### Candle

```python
@dataclass(frozen=True)
class Candle:
    symbol: str
    interval: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
```

`is_closed=False`이면 화면 표시만 합니다. 모델 입력에는 넣지 않습니다.

---

### AppState

```python
@dataclass
class AppState:
    mode: Literal["BOOTSTRAP", "LIVE", "PAUSED", "DEGRADED", "ERROR"]
    latest_closed_time: pd.Timestamp | None
    current_regime: str | None
    current_shock_type: str | None
    forecast_vol_1h: float | None
    target_exposure: float | None
    backtest_summary: dict
    model_status: dict
    data_status: dict
```

---

### VolatilityModel 인터페이스

나중에 GARCH 외 모델을 추가할 예정이므로 처음부터 model interface를 분리합니다.

```python
class VolatilityModel(Protocol):
    name: str

    def fit(self, returns: pd.Series) -> "FitResult":
        ...

    def forecast_one_step(self, returns: pd.Series) -> "VolForecast":
        ...
```

MVP에서는 하나만 구현합니다.

```text
GarchVolatilityModel
```

나중에 추가 가능한 모델은 다음입니다.

```text
EWMAVolatilityModel
HARVolatilityModel
RealizedVolatilityModel
RandomForestVolModel
LSTMVolModel
```

하지만 현재 범위에서는 **GARCH만 구현**합니다.

---

# 8. 실시간 루프 설계

## 8.1 시작 시점

앱 실행:

```bash
cmva
```

내부 동작:

```text
1. config 로드
2. local parquet 데이터 확인
3. 부족한 historical candle을 Binance REST로 보충
4. 데이터 검증
5. rolling features 초기 계산
6. GARCH 모델 fit
7. historical walk-forward backtest 실행
8. WebSocket 연결
9. TUI dashboard 표시
```

---

## 8.2 실행 중

WebSocket으로 candle 업데이트가 들어올 때마다:

```text
if candle.is_closed is False:
    update current candle display only

if candle.is_closed is True:
    save candle
    validate candle
    update returns
    update rolling volatility
    update rolling correlation
    update PCA1 share
    update dispersion
    update GARCH forecast
    detect volatility shock
    classify regime
    compute next exposure
    update paper backtest
    refresh dashboard
```

---

## 8.3 GARCH 재추정 주기

매시간 모든 종목의 GARCH를 완전 재추정하면 계산이 무거울 수 있습니다. 따라서 기본값은 다음이 좋습니다.

| 작업                       |                  주기 |
| ------------------------ | ------------------: |
| WebSocket candle display |                 실시간 |
| Closed candle 저장         |      candle close마다 |
| Feature update           |      candle close마다 |
| Forecast update          |      candle close마다 |
| Shock/regime update      |      candle close마다 |
| Paper backtest update    |      candle close마다 |
| GARCH refit              | 24개 closed candle마다 |
| GARCH 강제 refit           |            사용자가 `F` |
| Model selection 재실행      |          주 1회 또는 수동 |

즉:

```text
매시간 forecast는 갱신
매일 GARCH parameter refit
매주 model selection
```

이 구조가 실시간성과 계산 안정성의 균형이 좋습니다.

---

# 9. Backtesting은 어떻게 “실시간”으로 갱신되는가?

여기서 중요한 개념은 **historical backtest**와 **live paper backtest**를 구분하는 것입니다.

## 9.1 Historical backtest

앱 시작 시 과거 데이터 전체에 대해 walk-forward 방식으로 한 번 계산합니다.

```text
2024-01-01 ~ 현재까지
```

이 결과는 dashboard의 초기 성과표가 됩니다.

## 9.2 Live paper backtest

앱이 켜진 뒤에는 새 candle이 닫힐 때마다 한 step씩 추가합니다.

```text
t 시점:
  t까지의 데이터로 t+1 exposure 결정

t+1 candle close:
  t+1 실제 수익률 확인
  이전에 정한 exposure 적용
  비용 차감
  paper PnL 업데이트
```

즉, 실시간으로 백테스트가 “미래까지 계산되는” 것이 아니라, **새로운 실제 candle이 하나씩 생길 때마다 검증 결과가 누적**되는 것입니다.

이게 실무적으로 더 정직한 방식입니다.

---

# 10. 새 CMVA의 regime logic

## 10.1 입력 feature

```text
market_vol_t
forecast_vol_t
average_pairwise_corr_t
pca1_share_t
dispersion_t
shock_score_t
shock_breadth_t
```

## 10.2 regime label

```text
SYSTEMIC_RISK
ASSET_SPECIFIC
IDIOSYNCRATIC_HIGH_VOL
QUIET_CORRELATED
```

## 10.3 shock label

```text
NORMAL
MODERATE_SHOCK
IDIOSYNCRATIC_VOL_SHOCK
SYSTEMIC_VOL_SHOCK
VOL_REGIME_BUILDUP
```

## 10.4 decision logic

```text
if forecast_vol high
   and avg_corr high
   and pca1_share high:
       regime = SYSTEMIC_RISK

elif forecast_vol high
   and avg_corr not high
   and dispersion high:
       regime = IDIOSYNCRATIC_HIGH_VOL

elif forecast_vol low_or_mid
   and avg_corr low
   and pca1_share low:
       regime = ASSET_SPECIFIC

else:
       regime = QUIET_CORRELATED
```

Shock는 regime보다 더 짧은 이벤트입니다.

```text
if shock_score > severe_threshold
   and shock_breadth > 0.6
   and avg_corr high:
       shock_type = SYSTEMIC_VOL_SHOCK
```

---

# 11. 투자정책 함수

CMVA는 실거래 봇이 아니므로 실제 주문을 넣지 않습니다. 대신 paper position을 계산합니다.

```text
base_weight_t = target_vol_per_period / forecast_vol_t
```

Regime multiplier:

| Regime                   | Multiplier |
| ------------------------ | ---------: |
| `SYSTEMIC_RISK`          |     `0.20` |
| `IDIOSYNCRATIC_HIGH_VOL` |     `0.50` |
| `ASSET_SPECIFIC`         |     `1.00` |
| `QUIET_CORRELATED`       |     `0.70` |

최종 exposure:

```text
final_weight_t =
clip(base_weight_t * regime_multiplier_t, 0.0, max_leverage)
```

실현 수익률:

```text
net_return_{t+1}
= final_weight_t * basket_return_{t+1}
  - transaction_cost_bps * turnover_t / 10000
  - slippage_bps * turnover_t / 10000
```

절대 규칙:

```text
weight_t는 t까지의 데이터로 계산한다.
수익률은 t+1의 실제 return에 적용한다.
```

---

# 12. TUI 화면 구성

## 12.1 Dashboard

보여줄 항목:

```text
latest closed candle
current regime
current shock type
market forecast volatility
average pairwise correlation
PCA1 share
dispersion
target exposure
paper PnL
drawdown
model refit status
data connection status
```

---

## 12.2 Data 화면

```text
symbol별 latest candle
missing candle count
duplicate candle count
OHLC validation status
volume validation status
WebSocket status
REST fallback status
```

---

## 12.3 Features 화면

```text
rolling volatility
EWMA volatility
rolling average correlation
rolling BTC beta
PCA1 share
dispersion
```

---

## 12.4 Models 화면

```text
selected model by symbol
GARCH params
AIC/BIC
last refit time
forecast_vol_1h
standardized residual
shock score
```

---

## 12.5 Regime 화면

```text
current regime
regime timeline
shock status
regime thresholds
average exposure by regime
historical frequency by regime
```

---

## 12.6 Backtest 화면

```text
strategy cumulative return
equal-weight benchmark
BTC benchmark
naive volatility targeting benchmark
annualized volatility
Sharpe
max drawdown
turnover
cost impact
worst 1h return
```

---

## 12.7 Settings 화면

사용자가 바꿀 수 있는 값:

```text
symbols
interval
rolling window
target annual vol
max leverage
cost bps
slippage bps
GARCH refit frequency
shock threshold
regime threshold quantiles
```

단, 설정 변경 후에는 앱이 다음 중 하나를 묻습니다.

```text
Apply from now
Recompute historical backtest
Cancel
```

---

# 13. 구현 단계

## Phase 1 — Project skeleton + TUI shell

목표:

```text
cmva 실행 시 TUI 앱이 열린다.
아직 실제 모델은 없어도 화면 이동이 된다.
```

구현:

```text
pyproject.toml
README.md
cmva/__main__.py
cmva/app.py
cmva/tui/app.py
cmva/config.py
```

완료 기준:

```bash
python -m cmva
```

실행 시 dashboard가 열린다.

---

## Phase 2 — Candle data layer

목표:

```text
Binance REST로 historical 1h candle 수집
WebSocket으로 live kline 수신
closed candle만 저장
```

구현:

```text
data/rest_client.py
data/websocket_client.py
data/candle.py
data/storage.py
```

완료 기준:

```text
BTCUSDT, ETHUSDT, SOLUSDT 등 최소 5개 symbol의 1h candle 저장
current candle과 closed candle 구분
local parquet 저장
```

---

## Phase 3 — Data validation

목표:

```text
수집 데이터가 모델 입력으로 적합한지 검증
```

검증:

```text
timestamp continuity
duplicate candle
OHLC logic
volume >= 0
missing candle
outlier return
symbol coverage
```

완료 기준:

```text
Data 화면에서 symbol별 validation status 표시
validation_report.md export 가능
```

---

## Phase 4 — Feature engine

목표:

```text
closed candle 기준으로 rolling feature 계산
```

구현 feature:

```text
log return
basket return
realized volatility
EWMA volatility
rolling average pairwise correlation
rolling BTC beta
PCA1 explained variance share
dispersion
```

완료 기준:

```text
새 closed candle이 들어오면 feature가 incremental update됨
look-ahead bias test 통과
```

---

## Phase 5 — GARCH model MVP

목표:

```text
GARCH 기반 1h volatility forecast 생성
```

구현:

```text
models/base.py
models/garch.py
models/registry.py
forecast/volatility_forecaster.py
```

모델:

```text
GARCH(1,1)-Student-t
constant mean
```

처음에는 너무 많은 후보를 넣지 말고, 안정적인 baseline부터 완성합니다.

완료 기준:

```text
symbol별 forecast_vol_1h 표시
refit every 24 closed candles
사용자 F 입력 시 force refit
```

---

## Phase 6 — Shock + regime classifier

목표:

```text
현재 시장 상태를 shock와 regime으로 분류
```

구현:

```text
regime/shock.py
regime/classifier.py
regime/thresholds.py
```

완료 기준:

```text
Dashboard에 current regime 표시
Regime 화면에 regime timeline 표시
shock score와 shock breadth 표시
```

---

## Phase 7 — Policy + paper backtest

목표:

```text
forecast와 regime이 실제 리스크 조절에 도움이 되는지 검증
```

구현:

```text
policy/vol_target.py
policy/regime_vol_target.py
backtest/engine.py
backtest/live_paper.py
backtest/metrics.py
backtest/costs.py
```

벤치마크:

```text
BTC buy-and-hold
equal-weight buy-and-hold
naive volatility targeting
regime-aware volatility targeting
```

완료 기준:

```text
Backtest 화면에서 누적수익률, 변동성, Sharpe, MDD, turnover 표시
새 candle close마다 live paper result 업데이트
```

---

## Phase 8 — Report export

목표:

```text
면접/포트폴리오 제출 가능한 리포트 생성
```

구현:

```text
reports/markdown.py
reports/html.py
reports/plots.py
```

리포트 구성:

```text
1. Project overview
2. Data source
3. Data validation
4. Rolling features
5. GARCH forecast
6. Shock and regime classification
7. Backtest methodology
8. Performance
9. Limitations
10. Future work
```

---

## Phase 9 — Model 확장 대비

현재는 GARCH만 구현하지만, 나중에 다른 모델을 추가할 수 있도록 처음부터 registry 구조를 둡니다.

```python
MODEL_REGISTRY = {
    "garch": GarchVolatilityModel,
}
```

나중에:

```python
MODEL_REGISTRY = {
    "garch": GarchVolatilityModel,
    "ewma": EWMAVolatilityModel,
    "har": HARVolatilityModel,
    "rf": RandomForestVolatilityModel,
}
```

TUI의 Models 화면에서 모델을 바꾸는 기능은 미래 확장으로 둡니다.

---

# 14. 새 CMVA의 MVP 범위

너무 크게 만들면 완성이 어렵습니다. MVP는 다음으로 제한하는 게 좋습니다.

| 항목       | MVP 결정                                          |
| -------- | ----------------------------------------------- |
| 데이터      | Binance Spot public data                        |
| interval | 1h                                              |
| universe | 5~10개 USDT pair                                 |
| 실시간      | WebSocket kline stream                          |
| 모델       | GARCH(1,1)-Student-t                            |
| feature  | volatility, correlation, PCA1 share, dispersion |
| regime   | 4개 regime                                       |
| shock    | standardized residual 기반                        |
| 백테스트     | walk-forward paper backtest                     |
| UI       | Textual TUI                                     |
| 주문       | 없음                                              |
| API key  | 없음                                              |

---

# 15. Codex에 넣을 재구축 지시문

아래는 지금 요구사항을 반영한 새 개발 지시문입니다.

```text
You are Codex. Build a Python project named CMVA.

Project name:
CMVA — Crypto Market Volatility Analysis

Core product:
Build an interactive terminal application, not a collection of separate CLI commands.

The user should run:

  python -m cmva

or, after installation:

  cmva

This command opens a TUI application where the user can control the program using arrow keys, Enter, Space, and shortcut keys.

Do not design the primary workflow as:
  cmva collect
  cmva features
  cmva garch
  cmva backtest

Those operations should exist internally as services, not as separate commands the user must manually run.

Primary objective:
CMVA continuously collects public crypto market data, validates closed candles, computes rolling features, forecasts next-period volatility using GARCH, detects volatility shocks, classifies market regimes, computes a simulated target exposure, and updates a walk-forward paper backtest.

This is NOT a live trading bot.
Do NOT place real orders.
Do NOT require private API keys.
Use public market data only.
All portfolio decisions are simulated.

Data source:
Use Binance Spot public market data.

Historical bootstrap:
- Use Binance Spot REST klines endpoint for historical/recent candles.
- Use 1h interval.
- Store OHLCV candles locally.

Live updates:
- Use Binance Spot WebSocket kline stream.
- Display current open candle in the dashboard.
- Only persist and use candles when they are closed.
- Do not use an unfinished candle for feature generation, model fitting, forecast, regime classification, or backtesting.

Important candle rule:
If websocket kline payload indicates the candle is not closed, use it for display only.
If the candle is closed, it may enter the research pipeline.

Main app behavior:
On startup:
1. Load config.
2. Load local candle cache.
3. Fetch missing historical candles.
4. Validate data.
5. Compute initial features.
6. Fit or load GARCH model.
7. Run initial historical walk-forward backtest.
8. Start WebSocket data stream.
9. Open TUI dashboard.

During runtime:
1. Receive kline updates.
2. Update current candle display.
3. When a candle closes:
   - Save the candle.
   - Validate the new candle.
   - Update returns.
   - Update rolling volatility.
   - Update rolling correlation.
   - Update PCA1 share.
   - Update dispersion.
   - Update GARCH volatility forecast.
   - Detect volatility shock.
   - Classify market regime.
   - Compute next target exposure.
   - Settle previous paper-backtest step.
   - Refresh the dashboard.

Scheduling:
- Current candle display: on every websocket update.
- Closed candle processing: every closed 1h candle.
- Feature update: every closed 1h candle.
- Forecast update: every closed 1h candle.
- GARCH refit: every 24 closed candles by default.
- Force GARCH refit: user presses F.
- Historical backtest recompute: user presses B.
- Pause/resume live processing: user presses Space.
- Export report: user presses E.
- Quit: user presses Q.

Recommended stack:
- Python >= 3.11
- pandas
- numpy
- scipy
- arch
- statsmodels
- scikit-learn
- textual
- rich
- httpx or aiohttp
- websockets
- pyarrow
- pydantic
- pytest

Package layout:
cmva/
  __init__.py
  __main__.py

  app.py
  config.py
  state.py
  events.py
  logging_config.py

  tui/
    __init__.py
    app.py
    screens.py
    widgets.py
    bindings.py
    theme.py

  data/
    __init__.py
    rest_client.py
    websocket_client.py
    candle.py
    storage.py
    validation.py
    universe.py

  features/
    __init__.py
    returns.py
    volatility.py
    correlation.py
    pca.py
    dispersion.py
    rolling_beta.py

  models/
    __init__.py
    base.py
    garch.py
    registry.py
    diagnostics.py
    selection.py

  forecast/
    __init__.py
    volatility_forecaster.py
    portfolio_vol.py

  regime/
    __init__.py
    shock.py
    classifier.py
    thresholds.py

  policy/
    __init__.py
    base.py
    vol_target.py
    regime_vol_target.py

  backtest/
    __init__.py
    engine.py
    live_paper.py
    benchmarks.py
    costs.py
    metrics.py

  reports/
    __init__.py
    markdown.py
    html.py
    plots.py

  tests/
    test_candle_validation.py
    test_no_lookahead.py
    test_shock_classifier.py
    test_regime_classifier.py
    test_backtest_engine.py
    test_policy_shift.py

TUI screens:
1. Dashboard
2. Data
3. Features
4. Models
5. Regime
6. Backtest
7. Settings
8. Logs

Keyboard controls:
- Left/right arrows: switch tabs
- Up/down arrows: move selection
- Enter: select action
- Space: pause/resume
- R: refresh now
- F: force GARCH refit
- B: rerun historical backtest
- E: export report
- Q: quit

Candle definition:
Create a Candle dataclass with:
- symbol
- interval
- open_time
- close_time
- open
- high
- low
- close
- volume
- is_closed

Only is_closed=True candles are allowed into:
- storage as final data
- validation
- returns
- features
- GARCH fitting
- GARCH forecast
- shock detection
- regime classification
- backtest settlement

Rolling window definition:
A rolling window uses only the most recent N closed observations up to time t.
Example:
- 720 hourly candles = 30 days.
- A feature at time t must only use data through t.
- Never use t+1 data when computing a signal for t+1 return.

Feature engineering:
Compute:
- log return
- equal-weight basket return
- realized volatility
- EWMA volatility
- rolling average pairwise correlation
- rolling BTC beta
- PCA first component explained variance ratio
- dispersion

Definitions:
log_return_t = log(close_t / close_{t-1})
basket_return_t = equal-weight average return across valid symbols
dispersion_t = average_individual_volatility_t - basket_volatility_t
pca1_share_t = first PCA explained variance ratio over rolling return matrix

Forecast definition:
In CMVA, forecast means next-period volatility or variance forecast, not price direction.
The main forecast is:
  forecast_vol_{t+1|t}

GARCH MVP:
Implement GARCH(1,1) with Student-t residual distribution as the baseline model.
Use the arch package where possible.

GARCH equation:
r_t = mu + epsilon_t
epsilon_t = sigma_t * z_t
sigma2_t = omega + alpha * epsilon2_{t-1} + beta * sigma2_{t-1}

Output:
- asset forecast volatility
- basket forecast volatility
- standardized residual
- shock score

Model architecture:
Use a model interface so that future models can be added later.

class VolatilityModel:
    name: str
    def fit(self, returns): ...
    def forecast_one_step(self, returns): ...

For now, implement only:
- GarchVolatilityModel

But design registry:
MODEL_REGISTRY = {
    "garch": GarchVolatilityModel,
}

Future models should be easy to add but are out of scope now.

Volatility shock detection:
Implement shock score:

shock_score_t = abs(r_t - mu_forecast_t) / forecast_vol_t

Basic labels:
- NORMAL
- MODERATE_SHOCK
- IDIOSYNCRATIC_VOL_SHOCK
- SYSTEMIC_VOL_SHOCK
- VOL_REGIME_BUILDUP

Use:
- standardized shock score
- realized volatility jump ratio
- shock breadth
- average pairwise correlation
- dispersion

Example logic:
if shock_score is high and shock_breadth is high and avg_corr is high:
    shock_type = SYSTEMIC_VOL_SHOCK
elif shock_score is high and shock_breadth is low and dispersion is high:
    shock_type = IDIOSYNCRATIC_VOL_SHOCK
elif realized_vol_jump is high:
    shock_type = VOL_REGIME_BUILDUP
else:
    shock_type = NORMAL

Regime classification:
Classify each closed candle timestamp into:
1. SYSTEMIC_RISK
2. IDIOSYNCRATIC_HIGH_VOL
3. ASSET_SPECIFIC
4. QUIET_CORRELATED

Inputs:
- forecast_vol_t
- market_vol_t
- average_pairwise_corr_t
- pca1_share_t
- dispersion_t
- shock_type_t

Regime logic:
if forecast_vol high and avg_corr high and pca1_share high:
    regime = SYSTEMIC_RISK
elif forecast_vol high and avg_corr not high and dispersion high:
    regime = IDIOSYNCRATIC_HIGH_VOL
elif forecast_vol low_or_mid and avg_corr low and pca1_share low:
    regime = ASSET_SPECIFIC
else:
    regime = QUIET_CORRELATED

Use rolling or expanding quantile thresholds.
Do not use full-sample thresholds during walk-forward backtesting.

Policy:
Implement regime-aware volatility targeting.

base_weight_t = target_vol_per_period / forecast_vol_t

Regime multipliers:
- SYSTEMIC_RISK: 0.20
- IDIOSYNCRATIC_HIGH_VOL: 0.50
- ASSET_SPECIFIC: 1.00
- QUIET_CORRELATED: 0.70

final_weight_t = clip(base_weight_t * regime_multiplier_t, 0.0, max_leverage)

Backtest return:
net_return_{t+1}
= final_weight_t * basket_return_{t+1}
  - transaction_cost_bps * turnover_t / 10000
  - slippage_bps * turnover_t / 10000

No-look-ahead rule:
Every feature, forecast, regime label, shock label, and portfolio weight used for t+1 return must be computed using only data available through t.

Backtest modes:
1. Historical walk-forward backtest:
   - Runs on startup.
   - Uses historical closed candles.
   - Compares strategy against benchmarks.

2. Live paper backtest:
   - Updates one step whenever a new candle closes.
   - Uses the previously decided exposure.
   - Settles PnL after the next candle closes.
   - Does not place real orders.

Benchmarks:
- BTC buy-and-hold
- Equal-weight buy-and-hold
- Naive volatility targeting
- Regime-aware volatility targeting

Metrics:
- cumulative return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- Calmar ratio
- turnover
- total cost impact
- worst 1h return
- average exposure by regime
- return by regime
- drawdown during SYSTEMIC_RISK

Data validation:
Implement:
- timestamp continuity check
- duplicate candle check
- OHLC logic check
- volume >= 0 check
- missing candle detection
- outlier return detection
- symbol coverage summary

Do not forward-fill missing prices silently.

Storage:
Use local parquet files:
data/raw/
data/cleaned/
data/features/
data/models/
data/backtests/
reports/

Dashboard requirements:
Show:
- latest closed candle
- current WebSocket status
- current regime
- current shock type
- forecast basket volatility
- average pairwise correlation
- PCA1 share
- dispersion
- target exposure
- historical backtest summary
- live paper PnL
- model refit status
- data validation status

Reports:
User can press E to export:
- markdown report
- optional HTML report
- plots if implemented

Report must explicitly state:
- Public market data only.
- No private API keys.
- No live trading.
- All orders and positions are simulated.
- Closed candles only are used for research calculations.
- Signals are shifted by one period to avoid look-ahead bias.
- Transaction costs and slippage are included.
- GARCH forecast is a volatility forecast, not a price direction forecast.

Testing:
Use pytest.
Tests must not require internet.
Use synthetic candle data.

Required tests:
1. Unclosed candles are not used in feature calculation.
2. Duplicate timestamps are detected.
3. Invalid OHLC candles are rejected.
4. Rolling features do not use future data.
5. Shock classifier works on artificial examples.
6. Regime classifier assigns expected labels.
7. Backtest shifts weights by one period.
8. Transaction costs reduce returns when turnover is positive.
9. Live paper backtest settles previous exposure only after next candle closes.
10. GARCH failed fit is handled gracefully.

Acceptance criteria:
The project is complete when:
1. `python -m cmva` opens an interactive TUI.
2. The app can fetch historical 1h Binance Spot klines.
3. The app can receive live kline updates.
4. Only closed candles enter the research pipeline.
5. The dashboard displays latest candle, regime, shock status, forecast volatility, and target exposure.
6. Rolling features are computed without look-ahead bias.
7. GARCH(1,1)-Student-t produces 1h volatility forecasts.
8. Regime-aware volatility targeting policy produces simulated exposure.
9. Historical walk-forward backtest compares at least four strategies.
10. Live paper backtest updates after each new closed candle.
11. The user can pause/resume, force refit, rerun backtest, and export report from inside the TUI.
12. Unit tests pass.

Prioritize:
1. Correct time alignment.
2. Closed-candle discipline.
3. No-look-ahead backtesting.
4. Stable TUI operation.
5. Clear portfolio report.

Do not implement:
- real order placement
- private API key handling
- exchange account access
- leverage trading
- futures trading
- multi-model selection beyond GARCH in the MVP
```

---

## 최종 요약

CMVA는 이제 이렇게 정의하는 게 가장 좋습니다.

> **CMVA는 Binance 공개 암호화폐 candle 데이터를 수집하고, closed 1h candle마다 rolling volatility, correlation, PCA common-risk share, dispersion, GARCH volatility forecast를 갱신해 시장의 volatility shock과 regime을 판단하는 interactive terminal research app이다. 앱은 실제 주문을 넣지 않고, regime-aware volatility targeting 정책이 포트폴리오 리스크 조절에 도움이 되는지 historical walk-forward backtest와 live paper backtest로 검증한다.**

핵심 설계 원칙은 네 가지입니다.

```text
1. 사용자는 cmva 하나만 실행한다.
2. 진행 중인 candle은 화면 표시용이고, closed candle만 모델에 넣는다.
3. forecast는 가격 방향이 아니라 다음 1시간 변동성 예측이다.
4. backtest는 항상 signal을 한 칸 뒤로 밀어 look-ahead bias를 막는다.
```

[1]: https://textual.textualize.io/?utm_source=chatgpt.com "Textual Documentation"
[2]: https://www.statsmodels.org/dev/examples/notebooks/generated/rolling_ls.html?utm_source=chatgpt.com "Rolling Regression - statsmodels 0.15.0 (+1063)"
[3]: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints?utm_source=chatgpt.com "Market Data endpoints | Binance Open Platform"
[4]: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams?utm_source=chatgpt.com "WebSocket Streams | Binance Open Platform"
[5]: https://arch.readthedocs.io/en/latest/univariate/introduction.html?utm_source=chatgpt.com "Introduction to ARCH Models - arch 7.2.0 - Read the Docs"
