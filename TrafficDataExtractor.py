import os
def main():
    path = r"C:\Users\nitro\Downloads\Data\Extracted"
    filesToReadList = os.listdir(path) # returns a list of all the files in the passed path directory
    for file in filesToReadList:
        totalTraffic,totalData = readData(file, path)
        saveData(totalTraffic,totalData, file)

def readData(file, path):
    ## Reads the wikimedia data and gets the total traffic per every file.
    address = path + r'/' + file
    fileRead = open(address, "r")
    linesList = fileRead.readlines()
    fileRead.close()
    totalTraffic = 0
    totalData = 0
    for line in linesList:
        totalTraffic += int(line.split()[-2]) 
        totalData += int(line.split()[-1])
    return totalTraffic,totalData


def saveData(totalTraffic,totalData, file):
    ## Saves trafficData to a txt file
    fileSaveTo = open("TotalHourlyTraffic.csv","a")
    fileNameSplit = file.split("-")
    currDate = fileNameSplit[1] + "-" + fileNameSplit[2][0] + fileNameSplit[2][1] 
    currDateFormatted = currDate[0:4] + "-" +currDate[4:6] + "-" + currDate[6:]
    if checkIfDataExists(currDateFormatted):
        fileSaveTo.close()
        return False
    fileSaveTo.write(currDateFormatted + "," + str(totalTraffic) + ',' + str(totalData) + "\n")
    fileSaveTo.close()
    return True

def checkIfDataExists(date):
    ## If Data already exists dont overwrite
    existingLines = open("TotalHourlyTraffic.csv","r").readlines()
    for line in existingLines:
        lineList = line.split(',')
        if date == lineList[0]:
            return True

if __name__=='__main__':
    main()