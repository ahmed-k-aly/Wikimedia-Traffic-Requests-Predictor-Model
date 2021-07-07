from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.plotting import register_matplotlib_converters
from scipy.sparse import data
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
import pmdarima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()


def main():
    dataFrame = getData().Requests
    #dataFrame = normalizeData(dataFrame)
    newSARIMA(dataFrame)

def getTrainingTestingData(dataFrame):
    train_end = datetime(2009,9,1)
    test_end = datetime(2010,12,31)
    trainData = dataFrame[:train_end]
    testData = dataFrame[train_end:test_end]
    return trainData, testData



def normalizeData(dataset):
    avg = dataset.mean()
    dev = dataset.std()

    normalizedDataset = (dataset - avg) / dev
    
    return normalizedDataset


def deNormalizeData(dataset):
    avg = dataset.mean()
    dev = dataset.std()


    normalDS = (dataset * dev) + avg
    return normalDS

def visualizeData(dataFrame):
    plotDatas(dataFrame, 2009, 1, 1, 0, 2009, 12, 31, 23)
    dataFrame = dataFrame.diff().dropna()
    #plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
    #plt.plot(dataFrame)
    #plt.title(" Difference of Number of Requests")

    dataFrame = dataFrame.diff(7).dropna() # Seasonal Diff
    plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
    plt.plot(dataFrame)
    plt.title("Difference of Number of Requests after Seasonal Diff")

    plot_pacf(dataFrame)
    plot_acf(dataFrame)
    plt.show()
    return dataFrame


def newSARIMA(dataFrame):
    """
    Method that runs a SARIMA model on the passed dataset parameter 
    and outputs a predicted vs actual data graph
    """

    trainData, testData = getTrainingTestingData(dataFrame)
    #visualizeData(trainData)
    modelFit = runModel(trainData)
    #TODO: Add a robust anomaly detection algorithm, possibly STL algorithm
    predictions = getPredictions(testData, modelFit)
    #predictions = deNormalizeData(predictions)
    #dataFrame = deNormalizeData(dataFrame)
    residuals = testData - predictions # Value of errors
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/testData)),4))
    rolling_predictions = predictions
    counter = 0
    for train_end in testData.index:
        counter += 1
        if (counter % 1 == 0):
            pass
        else:
            continue
        train_data = dataFrame[:train_end]
        model = SARIMAX(train_data, order=(1,0,1), seasonal_order=(0,1,2,7))
        model_fit = model.fit()
        pred = model_fit.forecast()
        rolling_predictions[train_end] = pred


    makeFinalPlot(dataFrame, rolling_predictions)


def makeFinalPlot(dataFrame, predictions):
    plt.figure(figsize=(10,4))

    start_date = datetime(2009,1,1)
    end_date = datetime(2010,12,31)

    plt.plot(dataFrame)
    plt.plot(predictions)

    plt.legend(('Data', 'Predictions'), fontsize=16)

    plt.title('Daily Requests in Tens of Millions per hour', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    for day in range(start_date.day,end_date.day+1):
        plt.axvline(pd.to_datetime( str(start_date.year) + '-' + str(start_date.month) + '-' + str(day)) , color='k', linestyle='--', alpha=0.2)
    plt.show()






def runModel(dataFrame):
    """
    Method that creates a SARIMA model on the passed
    dataset and fits the model to the data.
    Returns the fitted model.
    """
    #orders = getSARIMA_order(dataFrame, 7)
    orders = [(1,0,1), (0,1,2,7)]
    model = SARIMAX(dataFrame, order= orders[0], seasonal_order = orders[1])
    modelFit = model.fit()
    print(modelFit.summary())
    return modelFit
    

def removeSeasonality(dataFrame):
    return dataFrame.diff(24).dropna()


def visualizeChange(dataFrame, nDiffs):
    dataFrame = removeTrend(dataFrame, nDiffs)
    plot_pacf(dataFrame)
    plot_acf(dataFrame)
    plt.show()



def removeTrend(dataFrame,nDiffs):
    kpss_diffs = ndiffs(dataFrame, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(dataFrame, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print("nDiffs: " + str(n_diffs))
    n_diffs = nDiffs
    if (n_diffs > 0):
        dataFrame = dataFrame.diff(n_diffs).dropna()
    return dataFrame


def stationarityTest(dataFrame, alpha = 0.05):
    kpss_pVal = pmdarima.arima.KPSSTest(dataFrame).should_diff(dataFrame)[0] 
    adf_pVal = pmdarima.arima.ADFTest(dataFrame).should_diff(dataFrame)[0]
    return max(adf_pVal, kpss_pVal) > alpha


def plotResiduals(residuals):
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.axhline(0, linestyle='--', color='k')
    plt.title('Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)



def plotDatas(data,
    start_year = 2008, start_month = 1, start_day = 1, start_hr=0, end_year=2010, end_month = 12, end_day= 5, end_hr = 7):
    """
    Method that plots the passed data with respect to the
    x-axis being between the passed starting dates and end dates.
    Paramerters are assigned the default values of 
    startDate(2008,01,01,00) to endDate(2008,01,05,07)
    """
    start_date = datetime(start_year,start_month,start_day,hour=start_hr)
    end_date = datetime(end_year,end_month,end_day,hour=end_hr)
    dataRange = data[start_date:end_date]
    plt.figure(figsize=(10,4))
    plt.plot(data)
    plt.title('Daily Wikimedia Requests in Tens of Millions', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    #for month in range(start_date.month,end_date.month+1):
     #   plt.axvline(pd.to_datetime(str(start_year) + '-' + str(month)) , color='k', linestyle='--', alpha=0.2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()


def parser(string):
    return datetime.strptime(string, '%Y-%m-%d-%H')


def getSARIMA_order(data, m):
    """
    Helper method that takes the data and the seasonal interval, m. 
    It returns THE SARIMA order as a list of
    Tuples of [(Arima Order)][(Seasonal order)]

    """
    model = auto_arima(data, m = m, seasonal = True,  error_action='ignore')
    
    return [model.order, model.seasonal_order]



def getPredictions(testData, modelFit):
    """
    Method that takes the testing data and the current
    model then returning the predicted data
    """
    predictions = modelFit.forecast(len(testData))
    return pd.Series(predictions, index=testData.index)



def getData():
    d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d-%H')
    df = pd.read_csv("new.csv", parse_dates = ['Date'], date_parser = d_parser)
    df = df.resample('D', on='Date').Requests.sum()
    df = pd.DataFrame({'Date':df.index, 'Requests':df.values})

    df.set_index('Date', inplace=True)
    idx = pd.date_range("2009-01-01", "2010-12-31", freq = 'D')    
    df = df.reindex(idx, method='nearest', fill_value=NaN)
    for value in df['Requests']:
        if value == 0:
            value = NaN
    return df

# def getData():
#     col_list = ["Date","Requests","DataSent"]
#     d_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %I-%p')

#     dataFrame = pd.read_csv("new.csv", parse_dates=[0], index_col = 0, squeeze=True, date_parser=parser)
    
#     dataFrame.asfreq(pd.infer_freq(dataFrame.index))
#     dataFrame.fillna(method='bfill')
#     # dataFrame.index = pd.DatetimeIndex(dataFrame.index)
#     print(dataFrame)
#     quit()
    # return dataFrame



if __name__ == "__main__":
    main()