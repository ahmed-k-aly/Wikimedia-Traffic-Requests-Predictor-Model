""" 
Runs a Seasonal Autoregressive Integrated Moving Average (SARIMA) model to forecast
2 years of wikimedia Traffic requests

"""

from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()


def main():
    startDate = datetime(2009, 1, 1)
    endDate = datetime(2010,12,31)
    dataFrame = getData().Requests
    dataFrame, predictions = newSARIMA(dataFrame, startDate, endDate)
    makeFinalPlot(dataFrame, predictions, startDate, endDate)

    

def getTrainingTestingData(dataFrame, startDate, endDate):
    """ 
    Splits the data into training and testing sets
    """
    train_end = startDate # no training data
    test_end = endDate 
    trainData = dataFrame[:train_end]
    testData = dataFrame[train_end:test_end]
    return trainData, testData


def visualizeData(dataFrame):
    """ 
    Function used to display acf and pacf graphs to determine sarima model parameters
    """
    plotDatas(dataFrame)
    dataFrame = dataFrame.diff().dropna()
    dataFrame = dataFrame.diff(7).dropna() # Seasonal Diff
    plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
    plt.plot(dataFrame)
    plt.title("Difference of Number of Requests after Seasonal Diff")

    plot_pacf(dataFrame)
    plot_acf(dataFrame)
    plt.show()
    return dataFrame


def newSARIMA(dataFrame, startDate, endDate):
    """
    Method that runs a SARIMA model on the passed dataset parameter 
    and outputs a predicted vs actual data graph
    """
    
    trainData, testData = getTrainingTestingData(dataFrame, startDate, endDate)
    #TODO: Add a robust anomaly detection algorithm, possibly STL algorithm
    counter = 0
    rolling_predictions = testData.copy()
    for train_end in testData.index:
        # loop over data
        counter += 1
        if (counter % 7 != 0): # Seasonality is weekly
            continue
        else:
            train_data = dataFrame[:train_end] # get training data
            model = SARIMAX(train_data, order=(1,0,1), seasonal_order=(0,1,2,7)) # set model parameters
            model_fit = model.fit() # fit model
            pred = model_fit.forecast(14) # forecast fourteen steps
            i = 0
            for prediction in pred:
                rolling_predictions[pred.index[i]] = prediction
                i+=1
    rolling_residuals = testData.copy()
    for i, value  in enumerate(testData):
        rolling_residuals[i] = value - rolling_predictions[testData.index[i]] # calculate residuals
    MAPE = round(np.mean(abs(rolling_residuals/testData)),4) # calculate Mean Absolute Percent Error
    print('Mean Absolute Percent Error:{}'.format(MAPE))
    plotResiduals(rolling_residuals) # plot residuals
    return dataFrame, rolling_predictions
    

def exportData(dataFrame, predictions):
    """ 
    Exports the dataFrame and the predictions into csvs
    """
    dataFrame.to_csv("RequestsData.csv") # save data
    predictions.to_csv("Predictions.csv") # save predictions
    

def makeFinalPlot(dataFrame, predictions, startDate, endDate):
    plt.figure(figsize=(10,4)) # figure size
    plt.plot(dataFrame) # plot data
    plt.plot(predictions) # plot predictions

    plt.legend(('Data', 'Predictions'), fontsize=16) # plot legend

    plt.title('Daily Requests in Tens of Millions per day', fontsize=20) # plot title
    plt.ylabel('Requests', fontsize=16) # plot y axis label
    for year in range(startDate.year, endDate.year+1): # sets vertical dashes per month
        for month in range(startDate.month,endDate.month+1):
            plt.axvline(pd.to_datetime(str(year) + '-' + str(month)) , color='k', linestyle='--', alpha=0.2)
    plt.axvline(pd.to_datetime(str(endDate.year+1) + '-' + str((endDate.month+1)%12 +1)) , color='k', linestyle='--', alpha=0.2)

    plt.show() # display plot     


def plotResiduals(residuals):
    """ 
    Plots the residuals of the predictions
    """
    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.axhline(0, linestyle='--', color='k')
    plt.title('Residuals from SARIMA Model', fontsize=20)
    plt.ylabel('Error', fontsize=16)



def plotDatas(data):
    """
    Method that plots the passed data with respect to the
    x-axis being between the passed starting dates and end dates.
    Parameters are assigned the default values of 
    startDate(2008,01,01,00) to endDate(2008,01,05,07)
    """
    plt.figure(figsize=(10,4))
    plt.plot(data)
    plt.title('Daily Wikimedia Requests in Tens of Millions', fontsize=20)
    plt.ylabel('Requests', fontsize=16)
    plt.axhline(0, color='k', linestyle='--', alpha=0.2)
    plt.show()


def parser(string):
    return datetime.strptime(string, '%Y-%m-%d-%H')

def getData():
    """ 
    Gets the data and cleans it to prepare for modeling.
    """
    d_parser = lambda x: datetime.strptime(x, '%Y-%m-%d-%H') # Parses the date and time from the data
    df = pd.read_csv("new.csv", parse_dates = ['Date'], date_parser = d_parser) # loads the csv file with the data
    df = df.resample('D', on='Date').Requests.sum() 
    df = pd.DataFrame({'Date':df.index, 'Requests':df.values})
    df.set_index('Date', inplace=True) # indexes the dataframe based on dates
    df = df.sort_values(by='Date') # sort values by date
    df = df.replace(to_replace=0, method='bfill') # correct for missing data using forwardfilling
    return df

if __name__ == "__main__":
    main()