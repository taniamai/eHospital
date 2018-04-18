import pyodbc
import pandas as pd
import numpy
import datetime
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

connection = pyodbc.connect(r'Driver={SQL Server};Server=DESKTOP-LI36RK3;'
                            r'Database=eHospital_NDGiaDinh;UID=sa;PWD=123456;Trusted_Connection=yes;')

dat = pd.read_sql("select ToaThuoc_Id, Duoc_Id, NgayTao, SoLuong "
                  "from dbo.ToaThuoc "
                  "where Duoc_Id in (SELECT Duoc_Id FROM dbo.DM_Duoc where TenHoatChat = 'aciclovir')",connection)

dat = pd.read_sql("select tt.ToaThuoc_Id, tt.Duoc_Id, kb.KhamBenh_Id, kb.ChanDoanICD_Id  "
                  "from dbo.ToaThuoc tt left join dbo.KhamBenh kb on tt.KhamBenh_Id = kb.KhamBenh_Id "
                  "where Duoc_Id in (SELECT Duoc_Id FROM dbo.DM_Duoc where TenHoatChat = 'aciclovir')",connection)

print(dat.shape)

datepsc = [datetime.datetime.strptime(str(d).strip().split(" ")[0], "%Y-%m-%d") for d in dat["NgayTao"]]
dat["ym"] = [d.strftime('%Y-%m') for d in datepsc]

df = pd.DataFrame(dat.groupby(["ym"]).size().reset_index())
df = df.drop(["ym"],axis=1)

plt.plot(df)
plt.show()
######################################################################################
dataset = df.values
dataset = dataset.astype('float32')

# LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default)
# or tanh activation functions are used. It can be a good practice to rescale the data to the range of 0-to-1
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1
look_back = 2
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# Batch size defines number of samples that going to be propagated through the network.

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print(pd.DataFrame(trainY[0], trainPredict[:,0]))
numpy.mean(abs(testY[0] - testPredict[:,0]))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()