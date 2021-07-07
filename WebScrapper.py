import functools
import html2text
import os
import sys
import threading
import requests
from bs4 import BeautifulSoup
from collections import deque
from TrafficDataExtractor import extractData
import concurrent.futures
import hashlib

hashDictionary = {}

def main():
    URLList = getURLs(2009, 2012)
    dataSetURLs = getDataSetURLs(URLList)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(getData, dataSetURLs)
    print("Done!")


def getData(url):
    count = 0
    while True:
        count += 1
        if count == 6: # Try the same url 5 times
            count = 0
            break
        try:
            fileName = url.split('/')[-1]
            downloadData(url)
            extractData(fileName)
            break
        except: # catch *all* exceptions
            pass

def removeFile(fileName):
    """ 
    Method that removes the downloaded file
    """
    try:
        if os.path.exists(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName)):
            os.remove(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName))
            return True
        else:
            print("The file does not exist")
            return False
    except:
        e = sys.exc_info()[0]
        if os.path.exists(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName)):
            os.remove(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName))
            return True
        else:
            print(e)
            return False


def downloadData(url, fileLocation='C:/Users/nitro/Downloads/Data/'):
    """ 
    Method that takes a url and a fileLocation. It then downloads the 
    url and saves it into the passed file location
    """
    fileName = url.split('/')[-1]
    fileNewAddress = fileLocation + fileName
    hash = getHash(fileName)
    while True:
        r = requests.get(url, allow_redirects=True)
        with open(fileNewAddress, 'wb') as f:
            f.write(r.content)
        hash = hashlib.md5(open(fileNewAddress,'rb').read()).hexdigest()
        if (hash == hashDictionary[fileName]):
            print("Perfect Hash Match")
            break
        else:
            print("Hashes Mismatch")
            os.remove(fileNewAddress)


def putHashesIntoDict(hashLink):
    page = requests.get(hashLink)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = soup.get_text().splitlines()
    for line in text:
        list = line.split("  ")
        list = list[::-1] # Reverse list
        x = dict([list])
        hashDictionary.update(x)


def getHash(fileName):
    return hashDictionary[fileName]        

def getDataSetURLs(URLList):
    dataSet = deque()
    for url in URLList:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        results = soup.findAll('a')
        for link in results:
            link = link.get('href')
            if link[-1] != 'z':
                continue
            finalLink = url + link
            dataSet.append(finalLink)
    return dataSet



def getURLs(firstYear, endYear):
    """ 
    Returns all the URLs from the first year (inclusive) to the end year
    (exclusive) in a list
    """
    
    list = deque()
    year = firstYear
    month = 9
    while (year < endYear):
        month = 1 + (month % 12)
        stringMonth = str(month)
        stringYear = str(year)
        if month < 10:
            stringMonth = '0' + str(month)
        url = "https://dumps.wikimedia.org/other/pagecounts-raw/" + stringYear + '/' + stringYear + '-' + stringMonth + '/'
        putHashesIntoDict(url + "md5sums.txt")
        year = year + 1 if month %12 == 0 else year 
        list.append(url)
    return list



if __name__ == "__main__":
    main()