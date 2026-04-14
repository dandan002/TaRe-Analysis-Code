import re
import datetime
import calendar
import numpy as np
import pandas as pd
from pathlib import Path
from xps.funcs import XPSMeas, XPSMeas_ion_milled


# Imports data from the SST measurement files. Automatically corrects for different counts due to numbers of sweeps.
# Imports multiple runs from the same file as different instances of XPSMeas
def importSST(path, nameDict, printv=True):
    path = Path(path)
    data = []
    xRayEnergy = []
    element = []
    fileCount = 0

    for filepath in sorted(path.glob("*.txt")):
        fileCount += 1
        file = filepath.name
        tempSampleName = nameDict[file.split('_')[0]]
        tempEnergy = file.split('_')[1]
        tempComment = 'Sample Code: ' + file.split('_')[0] + '  Index: ' + file.split('_')[2].split('.')[0]

        if tempEnergy not in xRayEnergy:
            xRayEnergy.append(tempEnergy)

        with open(filepath) as f:
            lines = f.readlines()

        regionList, infoList, dataList = [], [], []
        for ind in range(len(lines)):
            if lines[ind].find('[Region') >= 0:
                regionList.append(ind)
            if lines[ind].find('[Info ') >= 0:
                infoList.append(ind)
            if lines[ind].find('[Data') >= 0:
                dataList.append(ind)

        for ii in range(len(regionList)):
            tempElement = lines[regionList[ii] + 1].split('=')[1].rstrip('\n')
            if tempElement not in element:
                element.append(tempElement)
            tempSweeps = int(lines[infoList[ii] + 4].split('=')[1].rstrip('\n'))
            dataIndex = dataList[ii] + 1
            tempXData, tempYData, tempYErrData = [], [], []
            while lines[dataIndex] != '\n':
                tempXData.append(round(float(tempEnergy) - float(lines[dataIndex].split()[0]), 2))
                tempYData.append(float(lines[dataIndex].split()[1]))
                tempYErrData.append(np.sqrt(float(lines[dataIndex].split()[1])))
                dataIndex += 1
            data.append(XPSMeas(sample=tempSampleName, xrayEnergy=int(tempEnergy), element=tempElement,
                                comment=tempComment, sweeps=tempSweeps, BE=np.asarray(tempXData),
                                intensity=np.asarray(tempYData), intensityErr=np.asarray(tempYErrData)))

    if printv:
        print(f'\033[94mThe number of scans are \033[1m{fileCount} \033[0m\033[94m including all samples, Xray energies and element searches\033[0m')
        print('The Xray energies used are \033[94m' + ", ".join(xRayEnergy) + ' eV\033[0m')
        print('The scanned elements are: \033[94m' + ", ".join(element) + '\033[0m')

    return data, xRayEnergy, element, fileCount


def importIOS(filename):
    with open(filename) as f:
        lines = f.readlines()

    data = []

    sample = []
    sampleCount = 0
    xrayEnergy = []
    xrayEnergyCount = 0
    element = []
    elementCount = 0

    count = 0
    MeasLineList = []  # list of lines where a new sample measurement was saved in the file

    for line in lines:
        if line.find('# Group') >= 0:
            if line.find('Sample') >= 0:
                MeasLineList.append(count)
        count += 1

    print('\033[94m' + 'The number of sample measurements are {} '.format(len(MeasLineList)) + '\033[0m')
    MeasLineList.append(count)

    n = 0  # Count of XPS scan at for each sample at each power, xray energy and element

    for MeasIndex in range(len(MeasLineList) - 1):
        testVal = MeasLineList[MeasIndex]
        if len(lines[testVal].split()[3]) > 2:
            newSample = lines[testVal].split()[2] + ' ' + lines[testVal].split()[3][:-1]
        else:
            newSample = lines[testVal].split()[2] + ' ' + lines[testVal].split()[3]
        nameLength = len(lines[testVal].split())
        eVInd = lines[testVal].split().index('eV')
        measComment = ''
        if eVInd == nameLength - 1:
            measComment = '- as loaded'
        else:
            for indCom in range(eVInd + 1, nameLength):
                measComment = measComment + lines[testVal].split()[indCom] + ' '
        index = 0
        while (testVal + index < MeasLineList[MeasIndex + 1]):
            while (lines[testVal + index].find('#') == 0):
                if (lines[testVal + index].find('Region') == 2):
                    tempEnergyString = lines[testVal + index].split()[2]
                    newEnergy = re.search(r'\d+', tempEnergyString[-4:]).group()
                    elementIndex = tempEnergyString.index(newEnergy)
                    newElement = tempEnergyString[:elementIndex]
                    tempData = []
                if (lines[testVal + index].find('Number of Scans:') == 2):
                    newScanNum = int(re.search(r'\d+', lines[testVal + index]).group())
                index += 1
            dataExist = True
            while (dataExist):
                if (testVal + index < count):
                    tempData.append(lines[testVal + index].split())
                    index += 1
                    if testVal + index == count:
                        dataExist = False
                    elif lines[testVal + index].find('#') == 0:
                        dataExist = False
            plotXaxis = []
            plotYaxis = []
            YErr = []
            for k in range(len(tempData) - 1):
                plotXaxis.append(float(tempData[k][0]))
                plotYaxis.append(float(tempData[k][1]))
                YErr.append(np.sqrt(float(tempData[k][1])))
            n += 1
            data.append(XPSMeas(sample=newSample, xrayEnergy=int(newEnergy), element=newElement, comment=measComment,
                                sweeps=newScanNum, BE=np.asarray(plotXaxis), intensity=np.asarray(plotYaxis),
                                intensityErr=np.asarray(YErr)))

    print('\033[94m' + 'Total number of scans across all samples is {} '.format(n) + '\033[0m')

    return data


def importIOS_v2(filename):
    with open(filename) as f:
        lines = f.readlines()

    out = []
    BE = []
    intensity = []
    datestamp = None
    currentScan = ''

    atData = False
    for line in lines:
        if atData:
            if re.findall(r'\d', line) == [] or '#' in line:
                atData = False
                if not BE == []:
                    out.append(XPSMeas(sample=sample,
                                       xrayEnergy=int(xrayEnergy),
                                       element=element,
                                       BE=np.asarray(BE),
                                       intensity=np.asarray(intensity),
                                       comment='Scan: ' + currentScan,
                                       datestamp=datestamp))
                    BE = []
                    intensity = []

        else:
            if not '#' in line:
                atData = True

        if atData:
            BEend = line.index(' ')
            BE.append(float(line[:BEend]))

            intensityStart = line.rindex(' ')
            intensityEnd = line.rindex('\n')
            intensity.append(float(line[(intensityStart + 1):(intensityEnd)]))
        elif 'Group:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex('   ')
            sample = line[(startInd + 3):(finalInd)]
        elif 'Region:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex(' ')
            element = line[(startInd + 1):(finalInd)]
        elif 'Excitation Energy:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex(' ')
            xrayEnergy = float(line[(startInd + 1):(finalInd)])
        elif 'Scan:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex(' ')
            currentScan = line[(startInd + 1):(finalInd)]
        elif 'Number of Scans:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex(' ')
            scanNum = int(line[(startInd + 1):(finalInd)])
        elif 'Acquisition Date:' in line:
            finalInd = line.rindex('\n')
            startInd = line.index(':')
            substr = re.search(r'\d\d/\d\d/\d\d \d\d:\d\d:\d\d', line)
            substr = substr.group()

            # find datetime
            #####
            year = int(substr[6:8])
            month = int(substr[0:2])
            day = int(substr[3:5])

            hour = int(substr[9:11])
            minute = int(substr[12:14])
            second = int(substr[15:17])

            datestamp = datetime.datetime(year, month, day, hour, minute, second)
            #####

        elif not '#' in line:
            atData = True

    return out


def import_ThermoAlpha(filename):
    """
    Import XPS data from Avantage Excel files.

    Args:
        filename (str): Path to Excel file containing XPS data

    Returns:
        list: List of XPSMeas or XPSMeas_ion_milled objects containing the imported data.
              Returns XPSMeas_ion_milled objects if the scan has multiple etch levels.
    """
    # Extract excel using pandas
    excelfile = pd.ExcelFile(filename)
    dataOut = []
    Al_kAlpha_energy = 1486.295  # eV. X-ray energy of the source

    # loop over different samples (samples are in different sheets)
    for i, sheet in enumerate(excelfile.sheet_names):
        # Look for .VGD file name in first column to find data start
        firstColumn = excelfile.parse(sheet, usecols=[0]).values
        sampleNameRow = -1
        vgdPath = None
        for ii, name in enumerate(firstColumn):
            name = name[0]
            if isinstance(name, str):
                if name[-4:] == '.VGD':
                    sampleNameRow = ii
                    vgdPath = name
        if sampleNameRow == -1:
            continue

        # Extract sample name from path
        sampleText = excelfile.parse(sheet, skiprows=sampleNameRow + 1, usecols=[0], nrows=1)
        print(sampleText.keys()[0])

        # Check if this is an etched sample (multiple etch levels) by looking for "Depth Profile" in path
        isEtched = 'Depth Profile' in vgdPath if vgdPath else False

        # Extract sample name: For etched samples, get parent directory name before "Depth Profile"
        # For non-etched, get the parent directory name
        if isEtched and vgdPath:
            # Split path and find "Depth Profile", then use the directory before it
            pathParts = vgdPath.split('\\')
            depthIdx = -1
            for idx, part in enumerate(pathParts):
                if part == 'Depth Profile':
                    depthIdx = idx
                    break
            # Sample name is 2 directories before Depth Profile (e.g., .../ReTa04/Depth Profile/...)
            if depthIdx > 1:
                sampleName = pathParts[depthIdx - 1]
            else:
                sampleName = sampleText.keys()[0].split('\\')[-2]
        else:
            sampleName = sampleText.keys()[0].split('\\')[-2]

        # Parse element and comment from sheet name
        if ' ' in sheet:
            element = sheet.split(' ')[0]
            if re.search(r'\d+', sheet.split(' ')[-1]) is None:
                comment = ''
            else:
                comment = re.search(r'\d+', sheet.split(' ')[-1]).group()
        else:
            element = sheet
            comment = ''

        if isEtched:
            # Get etch time and level information by searching for the rows containing these labels
            fullSheet = excelfile.parse(sheet)

            # Search for rows containing "Etch Time" and "Etch Level"
            etchTimeRowIdx = None
            etchLevelRowIdx = None

            for idx in range(min(30, len(fullSheet))):  # Search first 30 rows
                row_vals = fullSheet.iloc[idx, :].values
                for val in row_vals:
                    if isinstance(val, str):
                        if 'Etch Time (EtchTime)' in val:
                            etchTimeRowIdx = idx
                        elif 'Etch Level (EtchLevel)' in val:
                            etchLevelRowIdx = idx

            if etchTimeRowIdx is None or etchLevelRowIdx is None:
                raise ValueError(f"Could not find etch time/level rows in sheet {sheet}")

            etchTimeRow = fullSheet.iloc[etchTimeRowIdx, :].values
            etchLevelRow = fullSheet.iloc[etchLevelRowIdx, :].values

            # Extract etch times and levels, starting from column 2 (column 0 is BE, column 1 is label)
            etchTimes = []
            etchLevels = []
            for col_idx in range(2, len(etchTimeRow)):
                etchTimeVal = etchTimeRow[col_idx]
                etchLevelVal = etchLevelRow[col_idx]
                # Check if values are numeric (not NaN or string labels)
                if isinstance(etchTimeVal, (int, float)) and isinstance(etchLevelVal, (int, float)):
                    if not np.isnan(etchTimeVal) and not np.isnan(etchLevelVal):
                        etchTimes.append(etchTimeVal)
                        etchLevels.append(int(etchLevelVal))

            # Get scan data for each etch level
            scan = excelfile.parse(sheet, skiprows=sampleNameRow + 8).to_numpy()

            # Create XPSMeas_ion_milled object for each etch level
            for etch_idx, (etchTime, etchLevel) in enumerate(zip(etchTimes, etchLevels)):
                # Column indices: 0 = BE, columns 2+ = intensity data for each etch level
                BE = scan[1:, 0]
                intensity = scan[1:, etch_idx + 2]  # etch_idx + 2 to skip BE and empty column

                dataOut.append(
                    XPSMeas_ion_milled(sample=sampleName, xrayEnergy=Al_kAlpha_energy, element=element,
                                      comment=comment, sweeps=None, BE=BE,
                                      intensity=intensity,
                                      intensityErr=np.sqrt(intensity),
                                      etchtime=etchTime, etchlevel=etchLevel)
                )
        else:
            # Non-etched sample: use standard XPSMeas object
            # Get scan data (BE and intensity from columns 0 and 2)
            scan = excelfile.parse(sheet, skiprows=sampleNameRow + 8, usecols=[0, 2]).to_numpy()

            dataOut.append(
                XPSMeas(sample=sampleName, xrayEnergy=Al_kAlpha_energy, element=element,
                        comment=comment, sweeps=None, BE=scan[1:, 0],
                        intensity=scan[1:, 1],
                        intensityErr=np.sqrt(scan[1:, 1]))
            )
    return dataOut


def importNeo(filepath, sample):
    # find datetime
    #####
    fileInd = filepath.rindex('/')
    filename = filepath[(fileInd + 1):]
    dateInd = filename.index('_')
    dateStr = filename[:dateInd]

    year = int(dateStr[0:4])
    monthStr = dateStr[5:8]
    month = list(calendar.month_abbr).index(monthStr)
    day = int(dateStr[9:11])

    timeStr = filename[(dateInd + 1):]
    hour = int(timeStr[0:2])
    minute = int(timeStr[3:5])
    second = int(timeStr[6:8])

    datestamp = datetime.datetime(year, month, day, hour, minute, second)
    #####

    # open file, read data
    with open(filepath) as f:
        lines = f.readlines()

    BE = []
    intensity = []
    sample = sample

    atData = False
    for line in lines:
        if atData:
            BEend = line.index(' ')
            BE.append(float(line[:BEend]))

            intensityStart = line.rindex(' ')
            intensityEnd = line.rindex('\n')
            intensity.append(float(line[(intensityStart + 1):(intensityEnd)]))

        elif 'Label:' in line:
            finalInd = line.rindex('\n')
            startInd = line.rindex(' ')
            element = line[(startInd + 1):(finalInd)]
        elif 'Analysis Source Energy:' in line:
            finalInd = line.rindex(' eV')
            startInd = line.rindex('  ')
            xrayEnergy = float(line[(startInd + 2):(finalInd)])
        elif '-------' in line:
            atData = True

    out = XPSMeas(sample=sample,
                  xrayEnergy=int(xrayEnergy),
                  element=element,
                  BE=np.asarray(BE),
                  intensity=np.asarray(intensity),
                  datestamp=datestamp)

    return out
