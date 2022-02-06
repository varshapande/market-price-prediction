#!/usr/bin/env python
# coding: utf-8

# # Step #1 Load the Data

# In[1]:


import math 
import numpy as np 
import pandas as pd 
from datetime import date, timedelta, datetime 
from pandas.plotting import register_matplotlib_converters 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates  
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns
today = date.today()
date_today = today.strftime("%Y-%m-%d")
stockname = 'NASDAQ'
date_start = '2016-01-04'

df=pd.read_csv(r'https://raw.githubusercontent.com/Fluid-AI/marketprophecy/main/NASDAQ%20Data/NASDAQ%20Training%20Data%20-%201st%20Jan%202016%20to%201st%20Jan%202022.csv',index_col=['Date'])
df.head()


# # Step #2 Exploring the Data

# In[2]:


# Plot line charts
df_plot = df.copy()

list_length = df_plot.shape[1]
ncols = 2
nrows = int(round(list_length / ncols, 0))

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
for i in range(0, list_length):
        ax = plt.subplot(nrows,ncols,i+1)
        sns.lineplot(data = df_plot.iloc[:, i], ax=ax)
        ax.set_title(df_plot.columns[i])
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.tight_layout()
plt.show()


# # Step #3 Preprocessing and Feature Selection

# In[3]:


# Indexing Batches
train_df = df.sort_values(by=['Date']).copy()
train_df


# In[4]:


date_index = train_df.index
date_index


# In[5]:


# Adding Month and Year in separate columns
d = pd.to_datetime(train_df.index)
d


# In[6]:


train_df['Month'] = d.strftime("%m")
train_df['Year'] = d.strftime("%Y") 


# In[7]:


train_df = train_df.reset_index(drop=True).copy()
train_df.head(5)


# In[8]:


# List of considered Features
FEATURES = ['High', 'Low', 'Open', 'Close',
            'Volume','Month']


# In[9]:


print([f for f in FEATURES])


# In[10]:


# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_df)
data_filtered = data[FEATURES]
data_filtered


# In[11]:


# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['Close']


# In[12]:


# Print the tail of the dataframe
data_filtered_ext.head()


# In[13]:


# Get the number of rows in the data
nrows = data_filtered.shape[0]
nrows


# In[14]:


# Convert the data to numpy values
np_data_unscaled = np.array(data_filtered)
np_data_unscaled.shape


# In[15]:


np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)


# In[16]:


np_data


# In[17]:


# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)


# In[18]:


# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(data_filtered_ext['Close'])
df_Close


# In[19]:


np_Close_scaled = scaler_pred.fit_transform(df_Close)


# In[20]:


sequence_length = 50


# In[21]:


index_Close = data.columns.get_loc("Close")


# In[22]:


train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)
train_data_len


# In[23]:


train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len - sequence_length:, :]


# In[24]:


def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    x = np.array(x)
    y = np.array(y)
    return x, y


# In[25]:


#Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)


# In[26]:


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# In[27]:


x_train


# In[28]:


print(x_train[1][sequence_length-1][index_Close])
print(y_train[0])


# # Step #4 Model Training

# In[29]:


# Configure the neural network model
model = Sequential()
n_neurons = x_train.shape[1] * x_train.shape[2]


# In[30]:


print(n_neurons, x_train.shape[1], x_train.shape[2])


# In[31]:


model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))) 
model.add(LSTM(n_neurons, return_sequences=False))
model.add(Dense(5))
model.add(Dense(1))


# In[32]:


# Compile the model
model.compile(optimizer='adam', loss='mse')


# In[33]:


# Training the model
epochs = 50
batch_size = 16
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(x_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs,
                    validation_data=(x_test, y_test)
                   )
                    
                    #callbacks=[early_stop])


# In[34]:


# Plot training & validation loss values
fig, ax = plt.subplots(figsize=(20, 10), sharex=True)
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
plt.legend(["Train", "Test"], loc="upper left")
plt.grid()
plt.show()


# # Step #5 Evaluate Model Performance

# In[35]:


# Get the predicted values
y_pred_scaled = model.predict(x_test)


# In[36]:


# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))


# In[37]:


# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')


# In[38]:


# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')


# In[39]:


# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

