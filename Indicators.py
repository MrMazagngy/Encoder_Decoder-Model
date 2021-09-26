class Indicators :
    """
        This Class Handles Different Types Of Indicators
    """
    def __init__(self , df):
        self.df = df

    def Moving_Average(self):

        # Simple 20-Day Moving Average Of Close Prices
        self.df['ma20'] = self.df['Close'].rolling(window=20).mean()
        self.df['ma20'][0:20] = self.df['ma20'][21]
        # Simple 50-Day Moving Average Of Close Prices
        self.df['ma50'] = self.df['Close'].rolling(window=50).mean()
        self.df['ma50'][0:50] = self.df['ma50'][51]
        # Simple 100-Day Moving Average Of Close Prices
        self.df['ma100'] = self.df['Close'].rolling(window=100).mean()
        self.df['ma100'][0:100] = self.df['ma100'][101]

        return

    def Bollinger_Bands(self):
        self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
        self.df['MA20'][0:20] = self.df['MA20'][21]
        # 20-Day Standard Deviation
        self.df['20dSTD'] = self.df['Close'].rolling(window=20).std()
        self.df['20dSTD'][0:20] = self.df['20dSTD'][21]
        # Upper & Lower Bands
        self.df['UpperBB'] = self.df['MA20'] + (self.df['20dSTD'] * 2)
        self.df['LowerBB'] = self.df['MA20'] - (self.df['20dSTD'] * 2)

    def MACD(self):
        k = self.df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        # 12-Day EMA of the closing price
        d = self.df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = k - d
        # 9-Day EMA of the MACD for the Trigger line
        macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_h = macd - macd_s
        self.df['macd'] = self.df.index.map(macd)
        self.df['macd_h'] = self.df.index.map(macd_h)
        self.df['macd_s'] = self.df.index.map(macd_s)