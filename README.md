# technical-analysis-lib
[![tests](https://github.com/chunyang666888/technical-analysis-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/chunyang666888/technical-analysis-lib/actions)


> A **typed, numpy-backed technical indicator toolkit** — SMA, EMA, MACD, RSI, Stochastic, KDJ, Bollinger Bands, ATR, OBV, **VWAP** and **Williams %R**. Pure functions that return `numpy` arrays (with `NaN` where undefined), so they drop straight into backtests, screeners, and dashboards. No pandas required.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)

## Why this repo exists

Indicator math is where quant candidates most often ship **silent bugs** (off-by-one windows, wrong smoothing, NaNs that poison a backtest). This library isolates each indicator behind a clean, tested function so the rest of your stack can trust the numbers — and shows interviewers you respect numerical correctness.

## Indicators

| Group | Function | Notes |
|-------|----------|-------|
| Trend | `sma`, `ema`, `macd` | EMA seeded at first value; MACD = (EMA12 − EMA26) with signal EMA9 |
| Momentum | `rsi`, `stochastic`, `kdj`, `williams_r` | Wilder's RSI; %K/%D; standard 2/3-smoothed KDJ; Williams %R ∈ [-100, 0] |
| Volatility | `bollinger_bands`, `atr` | Pop=0 std; Wilder-smoothed ATR |
| Volume | `on_balance_volume`, `vwap` | Signed cumulative volume; volume-weighted average price |

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Quick start

```python
import numpy as np
from indicators import sma, rsi, bollinger_bands

close = np.array([...])            # your close prices
print(sma(close, 20)[-1])          # 20-day simple MA
print(rsi(close)[-1])              # 14-day RSI
upper, mid, lower = bollinger_bands(close)
```

Run the bundled demo:

```bash
python examples/indicators_demo.py
```

## Design notes

- All functions accept any array-like and return `numpy.ndarray`.
- Leading entries are `NaN` until enough samples exist — never a fake zero.
- Indicators are side-effect free and vectorizable; compose them freely
  (e.g. `rsi` to filter entries, `atr` to size stops).

## Running tests

```bash
pytest -q
```

## Project structure

```
technical-analysis-lib/
├── indicators/
│   ├── __init__.py
│   ├── trend.py
│   ├── momentum.py
│   ├── volatility.py
│   ├── volume.py
│   ├── vwap.py
│   └── williams_r.py
├── examples/
│   └── indicators_demo.py
├── tests/
│   ├── test_trend.py
│   ├── test_momentum.py
│   ├── test_volatility.py
│   └── test_vwap_williams.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## License

MIT — free for personal and commercial use.
