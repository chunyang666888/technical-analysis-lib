"""Compute a basket of indicators on a synthetic OHLCV series (offline)."""
import numpy as np

from indicators import (
    sma, ema, macd,
    rsi, stochastic, kdj,
    bollinger_bands, atr,
    on_balance_volume,
)


def main():
    rng = np.random.default_rng(7)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 1, n))
    low = close - np.abs(rng.normal(0, 1, n))
    volume = rng.integers(1_000, 10_000, n)

    print("=== Trend ===")
    print("SMA(20) last :", round(float(sma(close, 20)[-1]), 2))
    print("EMA(20) last :", round(float(ema(close, 20)[-1]), 2))
    macd_line, signal, hist = macd(close)
    print(f"MACD last    : line={float(macd_line[-1]):.2f} signal={float(signal[-1]):.2f} hist={float(hist[-1]):.2f}")

    print("\n=== Momentum ===")
    print("RSI(14) last :", round(float(rsi(close)[-1]), 2))
    k, d = stochastic(high, low, close)
    print(f"Stoch last   : %K={float(k[-1]):.2f} %D={float(d[-1]):.2f}")
    K, D, J = kdj(high, low, close)
    print(f"KDJ last     : K={float(K[-1]):.2f} D={float(D[-1]):.2f} J={float(J[-1]):.2f}")

    print("\n=== Volatility ===")
    upper, mid, lower = bollinger_bands(close)
    print(f"BBands last  : U={float(upper[-1]):.2f} M={float(mid[-1]):.2f} L={float(lower[-1]):.2f}")
    print("ATR(14) last :", round(float(atr(high, low, close)[-1]), 2))

    print("\n=== Volume ===")
    print("OBV last     :", int(on_balance_volume(close, volume)[-1]))


if __name__ == "__main__":
    main()
