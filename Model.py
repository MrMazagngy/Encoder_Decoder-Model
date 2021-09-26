
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from Indicators import Indicators


# Import Stock Data
path = r'D:\Encoder/stocks/AAPL_2006-01-01_to_2018-01-01.csv'
with open(path, 'rb') as Stock_Data:
    df = pd.read_csv(Stock_Data)
    df.head()

# Get reproducible results everytime [Deterministic]
tf.random.set_seed(5)
np.random.seed(5) 

####################################################
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
"""
Apply Different Indicators to DataFrame
"""
Indicator= Indicators(df)

Indicator.Moving_Average()  # Apply Moving_Average
Indicator.Bollinger_Bands() # Apply Bollinger_Bands
Indicator.MACD() # Apply MACD


x_total = df[['Open' , 'High','Low' ,'Close', 'Volume' ,'ma100','UpperBB','LowerBB']] # Features
y_total = df[['Low']] # Target


x_total = x_total.fillna(df['Close'].mean())
y_total = y_total.fillna(df['Close'].mean())

y_total = y_total.to_numpy()

y_total = tf.expand_dims(y_total, -1)

"""
Preprocess Data so Model Can Use 10-Past Days To Predict The Next Day
"""
N = len(x_total)
T = 10 # Window
D = 8 # Number Of Features Used
X= []
Y = []

# Use the Past 10 Days to Predict Next Day
for t in range(len(x_total) - T):
  x = x_total[t:t+T]
  X.append(x)
  y = y_total[t+T]
  Y.append(y)

X = np.array(X).reshape(-1, T, D) # N x T x D
Y = np.array(Y)

"""
Splitting Data into Train (72%) , Test(10%) , Validate(18%)
"""
X_rest, X_test_1, y_rest, y_test_1 = train_test_split(X, Y, test_size=0.1, random_state=2 , shuffle=False)

X_train_1, X_validate, y_train_1, y_validate = train_test_split(X_rest, y_rest, test_size=0.2, random_state=2 , shuffle=False)

"""
Standardize Data using (StandardScaler)
"""
sc_x = StandardScaler()
# Standardize X Train Data
num_instances, num_time_steps, num_features = X_train_1.shape
X_train_1 = np.reshape(X_train_1, (-1, num_features))
X_train_1 = sc_x.fit_transform(X_train_1)
X_train_1 = np.reshape(X_train_1,(num_instances, num_time_steps, num_features))


# Standardize X Validate Data
num_instances, num_time_steps, num_features = X_validate.shape
X_validate = np.reshape(X_validate, (-1, num_features))
X_validate = sc_x.transform(X_validate)
X_validate = np.reshape(X_validate,(num_instances, num_time_steps, num_features))

# Standardize X Test Data
num_instances, num_time_steps, num_features = X_test_1.shape
X_test_1 = np.reshape(X_test_1, (-1, num_features))
X_test_1 = sc_x.transform(X_test_1)
X_test_1 = np.reshape(X_test_1,(num_instances, num_time_steps, num_features))


sc_y = StandardScaler()

# Standardize Y Data
y_train_1 = sc_y.fit_transform(y_train_1.reshape(-1,1))
y_validate = sc_y.transform(y_validate.reshape(-1,1))
y_test_1 = sc_y.transform(y_test_1.reshape(-1,1))



class Encoder_Decoder(tf.keras.models.Model):
    """
    Encoder-Decoder Model
    """

    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.layers.LSTM(100, return_state=True)
        self.decoder = tf.keras.layers.LSTM(100)
        self.dropout = tf.keras.layers.Dropout (.25)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs_encoder = self.encoder(inputs)
        encoder_states = outputs_encoder[1:]
        output_decoder = self.decoder(inputs, initial_state=encoder_states)
        output = self.dropout(output_decoder)
        output = self.dense(output)

        return output

"""
Train The Model
"""
ED_Model = Encoder_Decoder()
ED_Model.compile(loss="mean_squared_error",
                 optimizer=keras.optimizers.Adam() )
ED_Model.fit(X_train_1, y_train_1, validation_data=(X_validate, y_validate), epochs=30, batch_size=32, shuffle=False)

# Predict
ED_predictions = ED_Model.predict(X_test_1)

"""
Metrics Used To Evaluate The Model
"""
print("MSE", mean_squared_error(y_test_1, ED_predictions, squared=False)) # mean_squared_error Metric
print("MAE", mean_absolute_error(y_test_1, ED_predictions)) # mean_absolute_error Metric

# Plotting Target Data Vs Predictions
plt.figure(figsize=(10, 10))
plt.plot(y_test_1, label='Target')
plt.plot(ED_predictions, label='Prediction')
plt.title("Encoder-Decoder Forecast")
plt.ylabel('Low Price')
plt.xlabel('Time')
plt.show()