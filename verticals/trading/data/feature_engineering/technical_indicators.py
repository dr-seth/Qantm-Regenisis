"""
Technical Indicator Feature Extractor

ARY-1084: Feature Engineering Pipeline
Extracts 20+ technical analysis indicators using TA-Lib.

Created: 2026-02-17
"""

from typing import List
import pandas as pd
import numpy as np

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from .base import FeatureExtractor, FeatureMetadata


class TechnicalIndicatorExtractor(FeatureExtractor):
    """
    Extracts technical analysis indicators from OHLCV data.
    
    Implements 20+ indicators across categories:
    - Trend: SMA, EMA, MACD, ADX, Parabolic SAR
    - Momentum: RSI, Stochastic, Williams %R, CCI, MFI, ROC
    - Volatility: Bollinger Bands, ATR, Keltner Channels
    - Volume: OBV, VWAP, AD Line, CMF
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize technical indicator extractor.
        
        Args:
            prefix: Optional prefix for feature names
        """
        super().__init__(prefix)
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is required for TechnicalIndicatorExtractor. "
                "Install with: pip install TA-Lib"
            )
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with open, high, low, close, volume columns
        
        Returns:
            DataFrame with technical indicator features
        """
        self._validate_input(df)
        features = pd.DataFrame(index=df.index)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        volume = df['volume'].values.astype(float)
        
        # === TREND INDICATORS ===
        
        # Simple Moving Averages
        features[self._add_prefix('sma_10')] = talib.SMA(close, timeperiod=10)
        features[self._add_prefix('sma_20')] = talib.SMA(close, timeperiod=20)
        features[self._add_prefix('sma_50')] = talib.SMA(close, timeperiod=50)
        features[self._add_prefix('sma_200')] = talib.SMA(close, timeperiod=200)
        
        # Exponential Moving Averages
        features[self._add_prefix('ema_10')] = talib.EMA(close, timeperiod=10)
        features[self._add_prefix('ema_20')] = talib.EMA(close, timeperiod=20)
        features[self._add_prefix('ema_50')] = talib.EMA(close, timeperiod=50)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features[self._add_prefix('macd')] = macd
        features[self._add_prefix('macd_signal')] = macd_signal
        features[self._add_prefix('macd_hist')] = macd_hist
        
        # ADX - Average Directional Index
        features[self._add_prefix('adx')] = talib.ADX(high, low, close, timeperiod=14)
        features[self._add_prefix('plus_di')] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features[self._add_prefix('minus_di')] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Parabolic SAR
        features[self._add_prefix('sar')] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
        # Aroon
        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
        features[self._add_prefix('aroon_up')] = aroon_up
        features[self._add_prefix('aroon_down')] = aroon_down
        features[self._add_prefix('aroon_osc')] = talib.AROONOSC(high, low, timeperiod=14)
        
        # === MOMENTUM INDICATORS ===
        
        # RSI - Relative Strength Index
        features[self._add_prefix('rsi_14')] = talib.RSI(close, timeperiod=14)
        features[self._add_prefix('rsi_7')] = talib.RSI(close, timeperiod=7)
        features[self._add_prefix('rsi_21')] = talib.RSI(close, timeperiod=21)
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        features[self._add_prefix('stoch_k')] = slowk
        features[self._add_prefix('stoch_d')] = slowd
        
        # Stochastic RSI
        fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        features[self._add_prefix('stochrsi_k')] = fastk
        features[self._add_prefix('stochrsi_d')] = fastd
        
        # Williams %R
        features[self._add_prefix('willr')] = talib.WILLR(high, low, close, timeperiod=14)
        
        # CCI - Commodity Channel Index
        features[self._add_prefix('cci')] = talib.CCI(high, low, close, timeperiod=20)
        
        # MFI - Money Flow Index
        features[self._add_prefix('mfi')] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # ROC - Rate of Change
        features[self._add_prefix('roc_10')] = talib.ROC(close, timeperiod=10)
        features[self._add_prefix('roc_20')] = talib.ROC(close, timeperiod=20)
        
        # Momentum
        features[self._add_prefix('mom_10')] = talib.MOM(close, timeperiod=10)
        features[self._add_prefix('mom_20')] = talib.MOM(close, timeperiod=20)
        
        # Ultimate Oscillator
        features[self._add_prefix('ultosc')] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        # === VOLATILITY INDICATORS ===
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        features[self._add_prefix('bb_upper')] = bb_upper
        features[self._add_prefix('bb_middle')] = bb_middle
        features[self._add_prefix('bb_lower')] = bb_lower
        features[self._add_prefix('bb_width')] = (bb_upper - bb_lower) / bb_middle
        features[self._add_prefix('bb_pct')] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR - Average True Range
        features[self._add_prefix('atr_14')] = talib.ATR(high, low, close, timeperiod=14)
        features[self._add_prefix('atr_7')] = talib.ATR(high, low, close, timeperiod=7)
        
        # Normalized ATR (ATR / Close)
        features[self._add_prefix('natr')] = talib.NATR(high, low, close, timeperiod=14)
        
        # True Range
        features[self._add_prefix('trange')] = talib.TRANGE(high, low, close)
        
        # Keltner Channels (using ATR-based calculation)
        ema_20 = talib.EMA(close, timeperiod=20)
        atr_10 = talib.ATR(high, low, close, timeperiod=10)
        features[self._add_prefix('kc_upper')] = ema_20 + (2 * atr_10)
        features[self._add_prefix('kc_lower')] = ema_20 - (2 * atr_10)
        features[self._add_prefix('kc_pct')] = (df['close'] - (ema_20 - 2 * atr_10)) / (4 * atr_10)
        
        # === VOLUME INDICATORS ===
        
        # OBV - On Balance Volume
        features[self._add_prefix('obv')] = talib.OBV(close, volume)
        
        # AD Line - Accumulation/Distribution
        features[self._add_prefix('ad')] = talib.AD(high, low, close, volume)
        
        # ADOSC - Chaikin A/D Oscillator
        features[self._add_prefix('adosc')] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        # CMF - Chaikin Money Flow (manual calculation)
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        features[self._add_prefix('cmf')] = pd.Series(mfv, index=df.index).rolling(20).sum() / pd.Series(volume, index=df.index).rolling(20).sum()
        
        # VWAP - Volume Weighted Average Price (intraday)
        typical_price = (high + low + close) / 3
        features[self._add_prefix('vwap')] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # === PATTERN RECOGNITION (Binary) ===
        
        # Candlestick patterns (returns 100, 0, or -100)
        features[self._add_prefix('cdl_doji')] = talib.CDLDOJI(open_price, high, low, close) / 100
        features[self._add_prefix('cdl_hammer')] = talib.CDLHAMMER(open_price, high, low, close) / 100
        features[self._add_prefix('cdl_engulfing')] = talib.CDLENGULFING(open_price, high, low, close) / 100
        features[self._add_prefix('cdl_morningstar')] = talib.CDLMORNINGSTAR(open_price, high, low, close) / 100
        features[self._add_prefix('cdl_eveningstar')] = talib.CDLEVENINGSTAR(open_price, high, low, close) / 100
        
        # === DERIVED FEATURES ===
        
        # Price relative to moving averages
        features[self._add_prefix('close_sma20_ratio')] = df['close'] / features[self._add_prefix('sma_20')]
        features[self._add_prefix('close_sma50_ratio')] = df['close'] / features[self._add_prefix('sma_50')]
        features[self._add_prefix('close_ema20_ratio')] = df['close'] / features[self._add_prefix('ema_20')]
        
        # MA crossover signals
        features[self._add_prefix('sma_10_20_cross')] = (features[self._add_prefix('sma_10')] > features[self._add_prefix('sma_20')]).astype(int)
        features[self._add_prefix('sma_20_50_cross')] = (features[self._add_prefix('sma_20')] > features[self._add_prefix('sma_50')]).astype(int)
        features[self._add_prefix('ema_10_20_cross')] = (features[self._add_prefix('ema_10')] > features[self._add_prefix('ema_20')]).astype(int)
        
        # RSI zones
        features[self._add_prefix('rsi_oversold')] = (features[self._add_prefix('rsi_14')] < 30).astype(int)
        features[self._add_prefix('rsi_overbought')] = (features[self._add_prefix('rsi_14')] > 70).astype(int)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all technical indicator feature names."""
        names = [
            # Trend
            'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_10', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'plus_di', 'minus_di',
            'sar',
            'aroon_up', 'aroon_down', 'aroon_osc',
            # Momentum
            'rsi_14', 'rsi_7', 'rsi_21',
            'stoch_k', 'stoch_d',
            'stochrsi_k', 'stochrsi_d',
            'willr', 'cci', 'mfi',
            'roc_10', 'roc_20',
            'mom_10', 'mom_20',
            'ultosc',
            # Volatility
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
            'atr_14', 'atr_7', 'natr', 'trange',
            'kc_upper', 'kc_lower', 'kc_pct',
            # Volume
            'obv', 'ad', 'adosc', 'cmf', 'vwap',
            # Patterns
            'cdl_doji', 'cdl_hammer', 'cdl_engulfing', 'cdl_morningstar', 'cdl_eveningstar',
            # Derived
            'close_sma20_ratio', 'close_sma50_ratio', 'close_ema20_ratio',
            'sma_10_20_cross', 'sma_20_50_cross', 'ema_10_20_cross',
            'rsi_oversold', 'rsi_overbought'
        ]
        return [self._add_prefix(name) for name in names]
    
    def get_feature_metadata(self) -> List[FeatureMetadata]:
        """Get metadata for all technical indicator features."""
        metadata = [
            # Trend indicators
            FeatureMetadata('sma_10', 'Simple Moving Average (10 periods)', 'trend', 10),
            FeatureMetadata('sma_20', 'Simple Moving Average (20 periods)', 'trend', 20),
            FeatureMetadata('sma_50', 'Simple Moving Average (50 periods)', 'trend', 50),
            FeatureMetadata('sma_200', 'Simple Moving Average (200 periods)', 'trend', 200),
            FeatureMetadata('ema_10', 'Exponential Moving Average (10 periods)', 'trend', 10),
            FeatureMetadata('ema_20', 'Exponential Moving Average (20 periods)', 'trend', 20),
            FeatureMetadata('ema_50', 'Exponential Moving Average (50 periods)', 'trend', 50),
            FeatureMetadata('macd', 'MACD Line (12-26)', 'trend', 26),
            FeatureMetadata('macd_signal', 'MACD Signal Line (9)', 'trend', 35),
            FeatureMetadata('macd_hist', 'MACD Histogram', 'trend', 35),
            FeatureMetadata('adx', 'Average Directional Index (14)', 'trend', 14, value_range=(0, 100)),
            FeatureMetadata('plus_di', 'Plus Directional Indicator', 'trend', 14, value_range=(0, 100)),
            FeatureMetadata('minus_di', 'Minus Directional Indicator', 'trend', 14, value_range=(0, 100)),
            FeatureMetadata('sar', 'Parabolic SAR', 'trend', 1),
            FeatureMetadata('aroon_up', 'Aroon Up', 'trend', 14, value_range=(0, 100)),
            FeatureMetadata('aroon_down', 'Aroon Down', 'trend', 14, value_range=(0, 100)),
            FeatureMetadata('aroon_osc', 'Aroon Oscillator', 'trend', 14, value_range=(-100, 100)),
            # Momentum indicators
            FeatureMetadata('rsi_14', 'Relative Strength Index (14)', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('rsi_7', 'Relative Strength Index (7)', 'momentum', 7, value_range=(0, 100)),
            FeatureMetadata('rsi_21', 'Relative Strength Index (21)', 'momentum', 21, value_range=(0, 100)),
            FeatureMetadata('stoch_k', 'Stochastic %K', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('stoch_d', 'Stochastic %D', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('stochrsi_k', 'Stochastic RSI %K', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('stochrsi_d', 'Stochastic RSI %D', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('willr', 'Williams %R', 'momentum', 14, value_range=(-100, 0)),
            FeatureMetadata('cci', 'Commodity Channel Index', 'momentum', 20),
            FeatureMetadata('mfi', 'Money Flow Index', 'momentum', 14, value_range=(0, 100)),
            FeatureMetadata('roc_10', 'Rate of Change (10)', 'momentum', 10),
            FeatureMetadata('roc_20', 'Rate of Change (20)', 'momentum', 20),
            FeatureMetadata('mom_10', 'Momentum (10)', 'momentum', 10),
            FeatureMetadata('mom_20', 'Momentum (20)', 'momentum', 20),
            FeatureMetadata('ultosc', 'Ultimate Oscillator', 'momentum', 28, value_range=(0, 100)),
            # Volatility indicators
            FeatureMetadata('bb_upper', 'Bollinger Band Upper', 'volatility', 20),
            FeatureMetadata('bb_middle', 'Bollinger Band Middle', 'volatility', 20),
            FeatureMetadata('bb_lower', 'Bollinger Band Lower', 'volatility', 20),
            FeatureMetadata('bb_width', 'Bollinger Band Width', 'volatility', 20),
            FeatureMetadata('bb_pct', 'Bollinger Band %B', 'volatility', 20, value_range=(0, 1)),
            FeatureMetadata('atr_14', 'Average True Range (14)', 'volatility', 14),
            FeatureMetadata('atr_7', 'Average True Range (7)', 'volatility', 7),
            FeatureMetadata('natr', 'Normalized ATR', 'volatility', 14),
            FeatureMetadata('trange', 'True Range', 'volatility', 1),
            FeatureMetadata('kc_upper', 'Keltner Channel Upper', 'volatility', 20),
            FeatureMetadata('kc_lower', 'Keltner Channel Lower', 'volatility', 20),
            FeatureMetadata('kc_pct', 'Keltner Channel %', 'volatility', 20),
            # Volume indicators
            FeatureMetadata('obv', 'On Balance Volume', 'volume', 1),
            FeatureMetadata('ad', 'Accumulation/Distribution', 'volume', 1),
            FeatureMetadata('adosc', 'Chaikin A/D Oscillator', 'volume', 10),
            FeatureMetadata('cmf', 'Chaikin Money Flow', 'volume', 20, value_range=(-1, 1)),
            FeatureMetadata('vwap', 'Volume Weighted Average Price', 'volume', 1),
            # Patterns
            FeatureMetadata('cdl_doji', 'Doji Pattern', 'pattern', 1, value_range=(-1, 1)),
            FeatureMetadata('cdl_hammer', 'Hammer Pattern', 'pattern', 1, value_range=(-1, 1)),
            FeatureMetadata('cdl_engulfing', 'Engulfing Pattern', 'pattern', 2, value_range=(-1, 1)),
            FeatureMetadata('cdl_morningstar', 'Morning Star Pattern', 'pattern', 3, value_range=(-1, 1)),
            FeatureMetadata('cdl_eveningstar', 'Evening Star Pattern', 'pattern', 3, value_range=(-1, 1)),
            # Derived
            FeatureMetadata('close_sma20_ratio', 'Close/SMA20 Ratio', 'derived', 20),
            FeatureMetadata('close_sma50_ratio', 'Close/SMA50 Ratio', 'derived', 50),
            FeatureMetadata('close_ema20_ratio', 'Close/EMA20 Ratio', 'derived', 20),
            FeatureMetadata('sma_10_20_cross', 'SMA 10/20 Crossover', 'derived', 20, value_range=(0, 1)),
            FeatureMetadata('sma_20_50_cross', 'SMA 20/50 Crossover', 'derived', 50, value_range=(0, 1)),
            FeatureMetadata('ema_10_20_cross', 'EMA 10/20 Crossover', 'derived', 20, value_range=(0, 1)),
            FeatureMetadata('rsi_oversold', 'RSI Oversold Zone', 'derived', 14, value_range=(0, 1)),
            FeatureMetadata('rsi_overbought', 'RSI Overbought Zone', 'derived', 14, value_range=(0, 1)),
        ]
        
        # Add prefix to names
        for m in metadata:
            m.name = self._add_prefix(m.name)
        
        return metadata
