import os
import gzip
import sys

def main():
    pass


def extractData(fileName):    
    if os.path.exists(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName)):
        file = gzip.open(r'C:\Users\nitro\Downloads\Data\{}'.format(fileName),mode='rt')
    else:
        print("The file does not exist")
        return
    totalTraffic,totalData = readData(file, fileName)
    saveData(totalTraffic,totalData, fileName)
    print(fileName)


def readData(fileRead, fileName):
    """ 
    Reads the wikimedia data and gets the total traffic
    per every file.
    """
    linesList = fileRead.readlines()
    fileRead.close()
    removeFile(fileName)
    totalTraffic = 0
    totalData = 0
    for line in linesList:
        totalTraffic += int(line.split()[-2]) 
        totalData += int(line.split()[-1])
    return totalTraffic,totalData


def saveData(totalTraffic,totalData, file):
    ## Saves trafficData to a txt file
    fileToSaveTo = open("TotalHourlyTraffic.csv", 'a')
    fileNameSplit = file.split("-")
    currDate = fileNameSplit[1] + "-" + fileNameSplit[2][0] + fileNameSplit[2][1] 
    currDateFormatted = currDate[0:4] + "-" +currDate[4:6] + "-" + currDate[6:] # Gets the date of the file in the format YYYY-MM-DD-HR
    # if checkIfDataExists(currDateFormatted):
    #     fileSaveTo.close()
    #     return False
    fileToSaveTo.write(currDateFormatted + "," + str(totalTraffic) + ',' + str(totalData) + "\n") # Saves the data in the CSV
    fileToSaveTo.close()
    return True

def checkIfDataExists(date):
    ## If Data already exists don't overwrite
    with open("TotalHourlyTraffic.csv","r") as f:
        existingLines = f.readlines()
    for line in existingLines:
        lineList = line.split(',')
        if date == lineList[0]:
            return True



def removeFile(fileName):
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



if __name__=='__main__':
    main()