# Wikimedia-Traffic-Requests-Predictor-Model
> The goal of this project is to create a forecasting model that predicts wikiMedia workloads
### Method Used
> The forecasting model used in this project was a Seasonal Autoregressive Integrated Moving Average(SARIMA) model.
> The reason for SARIMA modeling is that since the data is a time-series, an autoregressive model seemed to best fit the traffic data, hence the AR part. The need to correct for the errors in our predictions resulted in choosing the MA component. Furthermore, our data exhibited stationarity, which resulted in the need to get rid of stationarity to better predict our data. Lastly, since the number of people accessing the data varies across the week, our data needed to account for this weekly seasonality. Thus, we decided that a SARIMA model should best fit our data
### ![pred](https://github.com/ahmed-k-aly/Wikimedia-Traffic-Requests-Predictor-Model/blob/master/Figures/Predictions.png)
#### ![residuals](https://github.com/ahmed-k-aly/Wikimedia-Traffic-Requests-Predictor-Model/blob/master/Figures/Residuals2.png)
