import requests
from bs4 import BeautifulSoup
from collections import deque

def main():
    URLList = getURLs(2008, 2009)
    dataSetURLs = getDataSetURLs(URLList)
    downloadData(dataSetURLs)
    

def downloadData(dataSetURLs, fileLocation='C:/Users/nitro/Downloads/Data/'):
    for url in dataSetURLs:
        r = requests.get(url, allow_redirects=True)
        print(url.split('/')[-1])
        with open(fileLocation + url.split('/')[-1], 'wb') as f:
             f.write(r.content)


def getDataSetURLs(URLList):
    dataSet = deque()
    URL = URLList.popleft()
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.findAll('a')
    for link in results:
        link = link.get('href')
        if link[-1] != 'z':
            continue
        finalLink = URL + link
        dataSet.append(finalLink)
        # with open('C:/Users/nitro/Downloads/Data/' + link, 'wb') as f:
        #      f.write(r.content)    
    return dataSet


def getURLs(firstYear, endYear):
    #returns all the URLs from the first year(inclusive) to the end year(exclusive) in a list 
    list = deque()
    year = firstYear
    month = 6
    while (year < endYear):
        month = 1 + (month % 12)
        stringMonth = str(month)
        stringYear = str(year)
        if month < 10:
            stringMonth = '0' + str(month)
        url = "https://dumps.wikimedia.org/other/pagecounts-raw/" + stringYear + '/' + stringYear + '-' + stringMonth + '/'
        year = year + 1 if month %12 == 0 else year 
        list.append(url)
    return list



if __name__ == "__main__":
    main()