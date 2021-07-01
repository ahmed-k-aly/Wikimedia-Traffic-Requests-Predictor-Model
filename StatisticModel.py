import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pmdarima.arima import ndiffs
import pmdarima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()


def main():
    dataFrame = getData()
    newSARIMA(dataFrame)

def getTrainingTestingData(dataFrame):
    train_end = datetime(2008,1,8, hour=0)
    test_end = datetime(2008,1,12, hour=1)
    trainData = dataFrame[:train_end]
    testData = dataFrame[train_end:test_end]
    return trainData, testData



def visualizeData(dataFrame):
    plotDatas(dataFrame, 2008, 1, 1, 0, 2008, 1, 12, 1)
    avg, dev = dataFrame.mean(), dataFrame.std()
    dataFrame = (dataFrame - avg) / dev
    plt.show()
    #dataFrame= np.log(dataFrame)
    dataFrame = dataFrame.diff().dropna()
    plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
    plt.plot(dataFrame)
    plt.title("Log Difference of Number of Requests")

    dataFrame = dataFrame.diff(24).dropna() # Seasonal Diff
    plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
    plt.plot(dataFrame)
    plt.title("Log Difference of Number of Requests after Seasonal Diff")

    plot_pacf(dataFrame)
    plot_acf(dataFrame)
    plt.show()
    return dataFrame


def newSARIMA(dataFrame):
    """
    Method that runs a SARIMA model on the passed dataset parameter 
    and outputs a predicted vs actual data graph
    """

    #visualizeData(dataFrame)
    trainData, testData = getTrainingTestingData(dataFrame)
    modelFit = runModel(trainData)
    #TODO: Add a robust anomaly detection algorithm, possibly STL algo
    predictions = getPredictions(testData, modelFit)
    residuals = testData - predictions # Value of errors
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/testData)),4))
    makeFinalPlot(dataFrame, predictions)


def makeFinalPlot(dataFrame, predictions):
    plt.figure(figsize=(10,4))

    start_date = datetime(2008,1,1,hour=0)
    end_date = datetime(2008,1,12,hour=1)

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
    model = SARIMAX(dataFrame, order= (1,1,1), seasonal_order = (3,1,3,24))
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


def getSARIMA_order(data):
    """
    Helper method that returns THE SARIMA order

    """
    model = auto_arima(data, m = 24, seasonal = True,  error_action='ignore')
    
    return [model.order, model.seasonal_order]



def getPredictions(testData, modelFit):
    """
    Method that takes the testing data and the current
    model then returning the predicted data
    """
    predictions = modelFit.forecast(len(testData))
    return pd.Series(predictions, index=testData.index)


def getData():
    col_list = ["Date","Number of Requests"]
    dataFrame = pd.read_csv("TotalHourlyTraffic.csv", parse_dates=[0], index_col = 0, squeeze=True, date_parser=parser)
    return dataFrame.asfreq(pd.infer_freq(dataFrame.index))




if __name__ == "__main__":
    main()