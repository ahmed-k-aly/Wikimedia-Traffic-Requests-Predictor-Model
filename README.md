# Wikimedia-Traffic-Requests-Predictor-Model
> The goal behind this project is to create a forecasting model that predicts wikiMedia workloads
## Methodology
<!-- Add Data collection section --> 
### Data Collection
> Data was collected from wikimedia dumps: https://dumps.wikimedia.org/other/pagecounts-raw/. The data was organized in a .gz file. Each file contained all the pages associated with the wikimedia domain and had how many requests and data were sent to and from each page in that hour. Thus, the average hour contained around 5,000,000 data points. 
> Data was downloaded in parallel using multithreading, then an md5 checksum was performed on the data to ensure that the data was downloaded correctly. In case, the hash returned a mismatch, which did happen a lot, the thread would keep trying to redownload the data until the hashes would match. 
> Once the data was downloaded, it'd be extracted and all the page requests would be summed and collected as the total number of requests for that hour. The data would then be extracted into the [TotalHourlyTraffic.csv](https://github.com/ahmed-k-aly/Wikimedia-Traffic-Requests-Predictor-Model/blob/c64746b348d9deeb207711f9febbeadcd4b9717a/TotalHourlyTraffic.csv) file.
hourly, which led to almost 22,000 data points for the two years of 2009 and 2010. Each day was downloaded as a .gz file that was then extracted and added into a csv file with all the 
### Modeling
> The forecasting model used in this project was a Seasonal Autoregressive Integrated Moving Average(SARIMA) model.
> The reason for SARIMA modeling is that since the data is a time-series, an autoregressive model seemed to best fit the traffic data, hence the AR part. The need to correct for the errors in our predictions resulted in choosing the MA component. Furthermore, our data exhibited stationarity, which resulted in the need to get rid of stationarity to better predict our data. Lastly, since the number of people accessing wikiMedia content varies across the week; for instance, more people might need to access wikiMedia sites during the working week than the weekend, so our data needed to account for this weekly seasonality. Thus, we decided that a SARIMA model with sesonality of seven days should best fit our data
###
### ![pred](https://github.com/ahmed-k-aly/Wikimedia-Traffic-Requests-Predictor-Model/blob/master/Figures/Predictions.png)
#### ![residuals](https://github.com/ahmed-k-aly/Wikimedia-Traffic-Requests-Predictor-Model/blob/master/Figures/Residuals2.png)
