import h5py as h5py
import numpy as np
from colorama import Fore, Style
import os
import re
import sys

class AuxClass:
    pass

class ProbeClass:
    pass

class StimClass:
    pass

class MeasurementListClass:
    pass

class DataClass:
    pass

    def addGroup(self, groupName):
        if "measurementList" in groupName:
            setattr(self, groupName, MeasurementListClass())
        else:
            print(Fore.RED + 'Please Add a Valid measurementListClass!')
            return

class MetaDataTagsClass:
    pass

class NirsClass:
    pass

    def addGroup(self, groupName):
        if "aux" in groupName:
            setattr(self, groupName, AuxClass())
        elif "probe" in groupName:
            setattr(self, groupName, ProbeClass())
        elif "stim" in groupName:
            setattr(self, groupName, StimClass())
        elif "data" in groupName:
            setattr(self, groupName, DataClass())
        elif "metaDataTags" in groupName:
            setattr(self, groupName, MetaDataTagsClass())
        else:
            print(Fore.RED + 'Please Add a Valid Group!')
            return

class SnirfClass:
    pass

    def addGroup(self, groupName):
        if 'nirs' in groupName:
            setattr(self, groupName, NirsClass())
        else:
            print(Fore.RED + 'Please Add a /Nirs Class!')
            return

def SnirfLoad(filePath):

    def getData(gID):
        # check actual data type and dimension, and print accordingly
        if h5py.check_string_dtype(gID.dtype):  # string
            if gID.len() == 1:
                data = gID[0].decode('ascii')
            else:
                data = []
                for y in gID:
                    data.append(y.decode('ascii'))
                data = np.array(data)
        else:
            data = gID[()]
        return data

    def buildDataset(oneClass, oneGroup):
        if isinstance(oneGroup, h5py.Group):
            for xx in oneGroup.keys():
                oneDataset = oneGroup[xx]
                if isinstance(oneDataset, h5py.Dataset):
                    data = getData(oneDataset)
                    setattr(oneClass, xx, data)
                else:
                    if 'measurementList' in xx:
                        setattr(oneClass, xx, MeasurementListClass())
                        newClass = getattr(oneClass, xx)
                        buildDataset(newClass, oneDataset)
                    else:
                        return
        return oneClass

    if ".snirf" in filePath:
        fileID = h5py.File(filePath, 'r')
    else:
        print(Fore.RED + "Please Input a Valid File (SNIRF)!")
        return

    # generate a SNIRF class
    oneSnirf = SnirfClass()
    for ii in fileID.keys():
        oneName = fileID[ii]
        if isinstance(oneName, h5py.Group):
            oneSnirf.addGroup(ii)
            oneNirs = getattr(oneSnirf, ii)
            for jj in oneName.keys():  # /nirs
                oneNirs.addGroup(jj)
                buildDataset(getattr(oneNirs, jj), oneName[jj])
        else:
            if h5py.check_string_dtype(oneName.dtype):
                setattr(oneSnirf, ii, oneName[0].decode('ascii'))
    return oneSnirf

def SnirfSave(snirfObject, filename, filePath,overWrite):

    def writeDataset(grp,groupObj,field):
        if isinstance(getattr(groupObj, field), str):
            grp.create_dataset(field, data=[eval('groupObj.' + field + ".encode('UTF-8')")])
        elif isinstance(getattr(groupObj, field), np.ndarray):
            if isinstance(getattr(groupObj, field)[0], str):
                grpField = getattr(groupObj, field)
                strArray = [grpField[i].encode('UTF-8') for i in range(grpField.size)]
                grp.create_dataset(field, data=strArray)
            else:
                grp.create_dataset(field, data=getattr(groupObj, field))
        else:
            grp.create_dataset(field, data=getattr(groupObj, field))
        return grp

    def writeGroup(f, groupObj, attribute):
        grp = f.create_group(attribute)

        for field in groupObj.__dict__.keys():
            if field[:2] != '__' and 'addGroup' not in field and 'Print' not in field:
                oneClass = getattr(groupObj, field)
                if hasattr(oneClass, '__dict__') or hasattr(oneClass, '__slots__'):
                    writeGroup(grp, getattr(groupObj, field), field)
                else:
                    grp = writeDataset(grp, groupObj, field)
        return f

    fileDirectory = filePath + filename + '.snirf'

    if hasattr(snirfObject, '__dict__') or hasattr(snirfObject, '__slots__'):
        if type(snirfObject).__name__ != 'SnirfClass':
            print(Fore.RED + 'Please input a Valid SNIRF Class Object!')
            return
        if os.path.isfile(fileDirectory):
            if overWrite == False:
                print(Fore.RED + 'File already Exist! Please input Another Filename! Otherwise set overWrite = True')
                return
    else:
        print(Fore.RED + 'Please input a Valid Class Object!')
        return

    with h5py.File(fileDirectory, 'w') as f:
        for attribute in snirfObject.__dict__.keys():
            if attribute[:2] != '__' and 'addGroup' not in attribute and 'Print' not in attribute:
                oneClass = getattr(snirfObject, attribute)
                if hasattr(oneClass, '__dict__') or hasattr(oneClass, '__slots__'):
                    f = writeGroup(f, getattr(snirfObject, attribute), attribute)
                else:
                    f = writeDataset(f, snirfObject, attribute)

def Validate(filePath):
    file = h5py.File(filePath, 'r')

    def getSpec(gID):
        # check spec dimension
        if "Pos2D" in gID.name or "Pos3D" in gID.name:
            specDim = 2
        elif "dataTimeSeries" in gID.name:
            if "aux" in gID.name:
                specDim = 1
            else:
                specDim = 2
        elif "measurementList" in gID.name:
            if "dataTypeLabel" in gID.name:
                specDim = 1
            else:
                specDim = 0
        elif "stim" in gID.name and "data" in gID.name:
            if "dataLabels" in gID.name:
                specDim = 1
            else:
                specDim = 2
        else:
            specDim = 1

        # check spec data type
        if "metaDataTags" in gID.name or 'formatVersion' in gID.name:
            specType = str
        elif "name" in gID.name or "Label" in gID.name:
            specType = str
        elif "Index" in gID.name:
            specType = int
        elif "dataType" in gID.name:
            specType = int
        else:
            specType = float

        return specType, specDim

    def getData(gID):
        # check actual data type and dimension, and print accordingly
        if h5py.check_string_dtype(gID.dtype):  # string
            actualDim = gID.ndim
            if gID.len() == 1:
                data = gID[0].decode('ascii')
                # msg = Fore.CYAN + '\t\tHDF5-STRING'
            else:
                data = []
                for y in gID:
                    data.append(y.decode('ascii'))
                data = np.array(data)
                # msg = Fore.CYAN + '\t\tHDF5-STRING 1D-Array'
        else:
            data = gID[()]
            if gID.ndim == 2:
                # msg = Fore.CYAN + '\t\tHDF5-FLOAT 2D-Array'
                actualDim = gID.ndim
            elif gID.ndim == 1:  # always a float
                dimension = gID.shape
                if dimension[0] == 1:
                    # if 'int' in gID.dtype.name:
                    #     msg = Fore.CYAN + '\t\tHDF5-Integer'
                    # else:
                    #     msg = Fore.CYAN + '\t\tHDF5-Single Float'
                    actualDim = 0
                else:
                    # msg = Fore.CYAN + '\t\tHDF5-FLOAT 1D-Array'
                    actualDim = gID.ndim
            elif gID.ndim == 0:
                # msg = Fore.CYAN + '\t\tHDF5-Integer'
                actualDim = gID.ndim
            else:
                return
        return actualDim, data

    def getAllNames(file):
        if isinstance(file, h5py.File):
            required = ["formatVersion", "nirs"]
            checkGroupChild(file, required)
        elif isinstance(file, h5py.Group):
            required = getRequiredDataset(file)
            checkGroupChild(file, required)
        elif isinstance(file, h5py.Dataset):
            completeDatasetList.append(file.name)
            CheckDataset(file)
        else:
            return 0

    def getOptional():
        optionalList = ["/nirs\d*/data\w*/measurementList\d*/wavelengthActual",
                        "/nirs\d*/data\w*/measurementList\d*/wavelengthEmissionActual",
                        "/nirs\d*/data\d*/measurementList\d*/dataTypeLabel",
                        "/nirs\d*/data\w*/measurementList\d*/sourcePower",
                        "/nirs\d*/data\w*/measurementList\d*/detectorGain",
                        "/nirs\d*/data\w*/measurementList\d*/moduleIndex",
                        "/nirs\d*/data\w*/measurementList\d*/sourceModuleIndex",
                        "/nirs\d*/data\w*/measurementList\d*/detectorModuleIndex",
                        "/nirs\d*/probe/wavelengthsEmission",
                        "/nirs\d*/probe/frequencies",
                        "/nirs\d*/probe/timeDelays",
                        "/nirs\d*/probe/timeDelayWidths",
                        "/nirs\d*/probe/momentOrders",
                        "/nirs\d*/probe/correlationTimeDelays",
                        "/nirs\d*/probe/correlationTimeDelayWidths",
                        "/nirs\d*/probe/sourceLabels",
                        "/nirs\d*/probe/detectorLabels",
                        "/nirs\d*/probe/landmarkPos2D",
                        "/nirs\d*/probe/landmarkPos3D",
                        "/nirs\d*/probe/landmarkLabels",
                        "/nirs\d*/probe/useLocalIndex",
                        "/nirs\d*/aux\d*/timeOffset",
                        "/nirs\d*/stim\d*/dataLabels",
                        "/nirs\d*/stim\d*",
                        "/nirs\d*/aux\d*"]
        return optionalList

    def checkSpecialCase(required, requiredIndex, child):
        if 'sourcePos2D' in child or 'detectorPos2D' in child:
            if 'sourcePos2D' not in required and 'detectPos2D' not in required:
                required.append("sourcePos2D")
                requiredIndex.append(0)
                required.append("detectorPos2D")
                requiredIndex.append(0)
            childForCheck = child
        elif 'sourcePos3D' in child or 'detectorPos3D' in child:
            if 'sourcePos3D' not in required and 'detectPos3D' not in required:
                required.append("sourcePos3D")
                requiredIndex.append(0)
                required.append("detectorPos3D")
                requiredIndex.append(0)
            childForCheck = child
        elif 'landmarkPos' in child:
            childForCheck = child
        else:
            childForCheck = ''.join(i for i in child if not i.isdigit())
        return required, requiredIndex, childForCheck

    def printEverything(gID, child, requireFlag):
        if requireFlag:
            if isinstance(gID[child], h5py.Dataset):
                if "stim" in gID.name or "aux" in gID.name:
                    print(Fore.MAGENTA + '\t' + gID.name + '/' + child)
                    print(Fore.GREEN + '\t\tRequired Dataset When Parent Group ' + gID.name + ' Presents')
                else:
                    print(Fore.MAGENTA + '\t' + gID.name + '/' + child)
                    print(Fore.GREEN + '\t\tRequired Dataset')
            if isinstance(gID[child], h5py.Group):
                print(Fore.MAGENTA + gID[child].name)
                print(Fore.GREEN + '\tRequired Indexed Group')
        else:
            OptionalFlag = False
            optionalList = getOptional()
            for x in optionalList:
                if re.match(x, gID[child].name):
                    if isinstance(gID[child], h5py.Dataset):
                        print(Fore.MAGENTA + '\t' + gID.name + '/' + child)
                        print(Fore.BLUE + '\t\tOptional Dataset')
                        OptionalFlag = True
                        break
                    if isinstance(gID[child], h5py.Group):
                        print(Fore.MAGENTA + gID[child].name)
                        print(Fore.BLUE + '\tOptional Indexed Group')
                        OptionalFlag = True
                        break
            if not OptionalFlag:
                if isinstance(gID[child], h5py.Dataset):
                    if 'metaDataTags' in gID.name:
                        print(Fore.MAGENTA + '\t' + gID.name + '/' + child)
                        print(Fore.YELLOW + '\t\tUser Defined optional Dataset')
                    else:
                        print(Fore.MAGENTA + '\t' + gID.name + '/' + child)
                        print(Fore.RED + '\t\tInvalid Dataset')
                        invalidDatasetNameList.append(gID.name)
                if isinstance(gID[child], h5py.Group):
                    print(Fore.MAGENTA + gID.name)
                    print(Fore.RED + '\tInvalid Indexed Group')
                    invalidGroupNameList.append(gID.name)

    def getRequiredDataset(gID):
        if 'measurementList' in gID.name:
            required = ["sourceIndex", "detectorIndex", "wavelengthIndex", "dataType", "dataTypeIndex"]
        elif 'data' in gID.name:
            required = ["dataTimeSeries", "time", "measurementList"]
        elif 'stim' in gID.name:
            required = ["name", "data"]
        elif 'aux' in gID.name:
            required = ["name", "dataTimeSeries", "time"]
        elif 'metaDataTags' in gID.name:
            required = ["SubjectID", "MeasurementDate", "MeasurementTime", "LengthUnit", "TimeUnit", "FrequencyUnit"]
        elif 'probe' in gID.name:
            required = ["wavelengths"]
        elif 'nirs' in gID.name:
            required = ["metaDataTags", "data", "probe"]
        else:
            return 0
        return required

    def checkGroupChild(gID, required):
        requiredIndex = [0] * len(required)
        for child in gID:
            requireFlag = False
            if any(chr.isdigit() for chr in child):
                [required, requiredIndex, childForCheck] = checkSpecialCase(required, requiredIndex, child)
            else:
                childForCheck = child
            if childForCheck in required:  # if child in RequiredField, change RequiredIndex
                requiredIndex[required.index(childForCheck)] = 1
                requireFlag = True
            #printEverything(gID, child, requireFlag)
            getAllNames(gID[child])
        if 0 in requiredIndex:  # check if requiredIndex has 0, if so, append the name
            for i in range(len(required)):
                if requiredIndex[i] == 0:
                    missingList.append(gID.name + '/' + required[i])

    def CheckDataset(gID):

        # check spec datatype and dimension
        specType, specDim = getSpec(gID)
        [actualDim, data] = getData(gID)

        if gID.dtype == 'int64' or gID.dtype == 'int32':
            actualType = int
        elif gID.dtype == 'uint64' or gID.dtype == 'uint32':
            actualType = int
        elif isinstance(data, str) or h5py.check_string_dtype(gID.dtype):
            actualType = str
        else:
            actualType = float

        if "metaDataTags" in gID.name and not h5py.check_string_dtype(gID.dtype):
            # implies an user defined field since all required datasets are string
            actualType = specType
            actualDim = specDim

        # compare actual and spec, and print out correct statement
        if actualType.__name__ != specType.__name__:
            # print(Fore.RED + '\t\tINVALID Data Type! Expecting: ' + str(specType.__name__) +
            #       '! But ' + str(np.dtype(gID.dtype.type)) + ' was given.')
            invalidDatasetTypeList.append(gID.name)
        if actualDim != specDim:
            # print(Fore.RED + '\t\tINVALID Data Dimension! Expecting: ' + str(specDim) +
            #       '! But ' + str(actualDim) + ' was given.')
            invalidDatasetDimList.append(gID.name)

    completeDatasetList = []

    # critical
    missingList = []
    invalidDatasetTypeList = []

    # warning
    invalidGroupNameList = []
    invalidDatasetNameList = []
    invalidDatasetDimList = []

    # validate starts
    getAllNames(file)

    # print validation details
    Decision = True
    # if np.size(invalidGroupNameList) > 0:
    #     print(Fore.YELLOW + "Warning!")
    #     print(Fore.YELLOW + "Invalid Group Detected: ")
    #     print(Fore.YELLOW + str(invalidGroupNameList) + '\n')
    if np.size(missingList) > 0:
        # print(Fore.RED + "Missing Required Dataset/Group Detected: ")
        # print(Fore.RED + str(missingList) + '\n')
        Decision = False
    # if np.size(invalidDatasetNameList) > 0:
    #     print(Fore.YELLOW + "Warning!")
    #     print(Fore.YELLOW + "Invalid Dateset Detected: ")
    #     print(Fore.YELLOW + str(invalidDatasetNameList) + '\n')
    if np.size(invalidDatasetTypeList) > 0:
        # print(Fore.RED + "Invalid Dataset Data Type Detected: ")
        # print(Fore.RED + str(invalidDatasetTypeList) + '\n')
        Decision = False
    # if np.size(invalidDatasetDimList) > 0:
    #     print(Fore.YELLOW + "Warning!")
    #     print(Fore.YELLOW + "Invalid Dataset Dimension Detected: ")
    #     print(Fore.YELLOW + str(invalidDatasetDimList) + '\n')

    print(Fore.WHITE + '----------------------------------')
    if Decision:
        print(Fore.GREEN + filePath + " is valid!")
    else:
        print(Fore.RED + filePath + " is invalid!")

    return Decision

def main():
    if sys.argv.__len__() > 1:
        filePath = sys.argv[1]
########################################################################
    # Load File into a SNIRF class given a directory
    aTestSnirfClass = SnirfLoad(filePath)

    # Validate given a directory and return a verbose


    # Validate given a defined SNIRF Class and return a verbose


    # Save Snirf file into another directory
    SnirfSave(snirfObject=aTestSnirfClass,
              filename='pysnirf2',
              filePath='/Users/andyzjc/Downloads/SeniorProject/SampleData/Homer3Example',
              overWrite=True)
########################################################################
    # Sanity check by reloading the saved SNIRF File

    # SavedSnirf = SnirfLoad()

    # Validate the saved snirf object

########################################################################
    # Create an empty Snirf class object and add fields into the class

    # Validate the new Snirf class object

    # Saved the new Snirf class object

