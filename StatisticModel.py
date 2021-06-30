from os import remove
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pandas.plotting import lag_plot
from pmdarima.arima import ndiffs
import pmdarima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()


def main():
    dataFrame = getData()
    newSARIMA(dataFrame)


def getTrainingTestingData(dataFrame):
    train_end = datetime(2008,7,10, hour=0)
    test_end = datetime(2008,7,12, hour=1)
    trainData = dataFrame[:train_end]
    testData = dataFrame[train_end:test_end]
    return trainData, testData


def newSARIMA(dataFrame):
    plotDatas(dataFrame,2008, 7, 1, 0, 2008, 8, 12, 1)
    trainData, testData = getTrainingTestingData(dataFrame)
    newData = cleanData(trainData)
    modelFit = runModel(newData)
    
    predictions = getPredictions(testData, modelFit)
    residuals = testData - predictions # Value of errors
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/testData)),4))
    makeFinalPlot(dataFrame, predictions)


def makeFinalPlot(dataFrame, predictions):
    plt.figure(figsize=(10,4))

    start_date = datetime(2008,7,10,hour=0)
    end_date = datetime(2008,7,12,hour=1)

    plt.plot(dataFrame)
    plt.plot(predictions)

    plt.legend(('Data', 'Predictions'), fontsize=16)

    plt.title('Daily Requests in Tens of Millions per hour', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    for day in range(start_date.day,end_date.day+1):
        plt.axvline(pd.to_datetime( str(start_date.year) + '-' + str(start_date.month) + '-' + str(day)) , color='k', linestyle='--', alpha=0.2)
    plt.show()



def runModel(dataFrame):
    plot_pacf(dataFrame)
    plot_acf(dataFrame)
    plt.show()
    # AR = 4
    # MA = 2    
    # MAPE = 4.74%
    # Get Orders
    orders = getSARIMA_order(dataFrame)
    print(orders)
    model = SARIMAX(dataFrame, order= (2,0,2), seasonal_order= (1,1,1))
    modelFit = model.fit()
    print(modelFit.summary())
    return modelFit
    # Create Model
    # Fit Model
    # Make Forecast
    # Get Residuals
    

def removeSeasonality(dataFrame):
    return dataFrame.diff(24).dropna()



def cleanData(dataFrame):
    isStationary = stationarityTest(dataFrame)
    #if isStationary:
    
    #dataFrame = removeTrend(dataFrame)
    # remove trend
    # remove seasonality
    # transform data back
    return dataFrame
    # return data


def removeTrend(dataFrame):
    kpss_diffs = ndiffs(dataFrame, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(dataFrame, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print("nDiffs: " + str(n_diffs))
    n_diffs = 1
    if (n_diffs > 0):
        dataFrame = dataFrame.diff(n_diffs).dropna()
    return dataFrame


def stationarityTest(dataFrame, alpha = 0.05):
    kpss_pVal = pmdarima.arima.KPSSTest(dataFrame).should_diff(dataFrame)[0] 
    adf_pVal = pmdarima.arima.ADFTest(dataFrame).should_diff(dataFrame)[0]
    return max(adf_pVal, kpss_pVal) > alpha



def sarimaModel(dataset):
    """
    Method that runs a SARIMA model on the passed dataset parameter 
    and outputs a predicted vs actual data graph
    """

    train_end = datetime(2008,1,4, hour=0)
    test_end = datetime(2008,1,5, hour=8)
    trainData = dataset[:train_end]
    testData = dataset[train_end:test_end]

    allOrds= getSARIMA_order(trainData)
    order, SeasonalOrder = allOrds[0], allOrds[1] 

    model = SARIMAX(trainData, order=order, seasonal_order=SeasonalOrder)
    
    modelFit = model.fit()
    print(modelFit.summary())

    predictions = getPredictions(testData, modelFit)
    residuals = testData - predictions # Value of errors
    

    plt.figure(figsize=(10,4))

    start_date = datetime(2008,1,1,hour=0)
    end_date = datetime(2008,1,5,hour=8)

    plt.plot(dataset)
    plt.plot(predictions)

    plt.legend(('Data', 'Predictions'), fontsize=16)

    plt.title('Daily Requests in Tens of Millions per hour', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    for day in range(start_date.day,end_date.day+1):
        plt.axvline(pd.to_datetime( str(start_date.year) + '-' + str(start_date.month) + '-' + str(day)) , color='k', linestyle='--', alpha=0.2)
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/testData)),4))
    plt.show()



def plotDatas(data,
    start_year = 2008, start_month = 1, start_day = 1, start_hr=0, end_year=2008, end_month = 1, end_day= 5, end_hr = 7):
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
    plt.plot(dataRange)
    plt.title('Daily Wikimedia Requests in Tens of Millions', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    for day in range(start_date.day,end_date.day+1):
        plt.axvline(pd.to_datetime(str(start_year) + '-' + str(start_month) + '-' + str(day)) , color='k', linestyle='--', alpha=0.2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()

def parser(string):
    return datetime.strptime(string, '%Y-%m-%d-%H')


def getSARIMA_order(trainData):
    """
    Helper method that returns THE SARIMA order

    """
    model = auto_arima(trainData, m = 24, seasonal = True,  error_action='ignore')
    
    return [model.order, model.seasonal_order]


def getSeasonalOrder(P = 1,D = 1,Q = 1, m = 1):
    """
    Helper method that returns the seasonal order
    all parameters are assigned default values of 1 

    """
    return (P,D,Q, m)


def getPredictions(testData, modelFit):
    """
    Method that takes the testing data and the current
    model then returning the predicted data
    """
    predictions = modelFit.forecast(len(testData))
    return pd.Series(predictions, index=testData.index)




def makePACF(dataset, numLags = 15):
    """
    Method that runs a PACF on the passed dataset
    """
    pacfVals = pacf(dataset, nlags=numLags)
    plt.bar(range(numLags), pacfVals[:numLags])  
    plt.show()


def makeACF(dataset, numLags = 15):
    """
    Method that runs a PACF on the passed dataset
    """
    acfVals = acf(dataset)
    plt.bar(range(numLags), acfVals[:numLags])
    print(acfVals)
    plt.show()


def getData():
    col_list = ["Date","Data Sent"]
    dataFrame = pd.read_csv("TotalHourlyTraffic2.csv", usecols = col_list, parse_dates=[0], index_col = 0, squeeze=True, date_parser=parser)
    print(dataFrame)
    return dataFrame.asfreq(pd.infer_freq(dataFrame.index))




if __name__ == "__main__":
    main()