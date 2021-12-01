import unittest
import src.pysnirf2
from src.pysnirf2 import Snirf, NirsElement, StimElement, MetaDataTags
import h5py
import os
import sys
import time
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping
import numpy as np

VERBOSE = True  # Additional print statements in each test
SNIRF_DIR = 'data'  # Sample data source
WD = 'wd'  # Working directory for testing

ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)
def getsize(obj_0):
    """
    Recursively calculate size of object & members in bytes.
    the work of Aaron Hall https://stackoverflow.com/a/30316760
    """
    _seen_ids = set()
    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            if getattr(obj, 'items') is not None:
                size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def compare_snirf(snirf_path_1, snirf_path_2, enforce_length=True):
    '''
    Returns a struct pairing the Dataset names appearing in each SNIRF file with
    the result of comparing these values after a cast to numpy array. If None,
    the Dataset appears in only one of the two SNIRF files.
    '''
    f1 = h5py.File(snirf_path_1, 'r')
    f2 = h5py.File(snirf_path_2, 'r')
    loc_f1 = get_all_dataset_locations(f1)
    loc_f2 = get_all_dataset_locations(f2)
    
    if VERBOSE:
        print('Loaded', f1, 'and', f2, 'for comparison')
    
    results = {}
    
    locations = list(set(loc_f1 + loc_f2))
    for location in locations:
        if not (location in loc_f1 and location in loc_f2):
            results[location] = None
        else:
            arr1 =  np.array(f1[location])
            arr2 =  np.array(f2[location])
            if not enforce_length:
                if arr1.size == 1 or len(arr1) == 1:
                    try:
                        arr1 = arr1[0]
                    except IndexError:
                        arr1 = arr1[()]
                if arr2.size == 1 or len(arr2) == 1:
                    try:
                        arr2 = arr2[0]
                    except IndexError:
                        arr2 = arr2[()]
            eq = arr1 == arr2
            if not (type(eq) is np.bool_ or type(eq) is bool):
                eq = eq.all()
            if VERBOSE and not eq:
                print(type(arr1), arr1, '!=', type(arr2), arr2)
            results[location] = bool(eq)
    return results

        
def get_all_dataset_locations(h):
    '''
    Recursively create list of all relative names of the Datasets in an HDF5
    group or file
    '''
    locations = []
    for key in h.keys():
        if type(h[key]) is h5py.Dataset:
            locations.append(h[key].name)
        else:
            locations += get_all_dataset_locations(h[key])
    return locations
        

def dataset_equal_test(test, fname1, fname2):
    """
    Asserts that the HDF files at fname1 and fname2 have the same datasets
    """
    result = compare_snirf(fname1, fname2)
    false_keys = []
    none_keys = []
    for key in result.keys():
        if result[key] is None:
            none_keys.append(key)
        elif result[key] is False:
            false_keys.append(key)
        elif result[key] is True:
            pass
        else:
            raise ValueError('test_loading_saving failed. Invalid comparison between ', fname1 + ' and ' + fname2 + 'resulted in value' + str(result[key]))
    if len(none_keys) > 0 and VERBOSE:
        print('The following keys were not found in both SNIRF files:')
        print(none_keys)
    if len(false_keys) > 0 and VERBOSE:
        print('The following keys were not equivalent:')
        print(false_keys)
    missing = np.array([('metaDataTags' in key or 'stim0' in key or 'measurementList0' in key or 'data0' in key or 'aux0' in key or 'nirs0' in key) for key in none_keys]).astype(bool)
    test.assertTrue(missing.all(), msg=fname1 + ' and ' + fname2 + 'not equal: specified datasets are missing from the copied file: ' + str(none_keys))
    test.assertFalse(len(false_keys) > 0, msg=fname1 + ' and ' + fname2 + 'are not equal: datasets were incorrectly copied: ' + str(false_keys))

def _print_keys(group):
    for key in group.keys():
        print(key)


# -- Tests --------------------------------------------------------------------

class TestSnirf(unittest.TestCase):
    
    def test_dynamic(self):
        """
        Confirm that dynamically loaded files have smaller memory footprints
        and faster load-times than non dynamically loaded files
        """
        times = [-1, -1]
        sizes = [-1, -1]
        for i, mode in enumerate([True, False]):
            s = []
            start = time.time()
            for file in TEST_FILES:
                s.append(Snirf(file, dynamic_loading=mode))
            times[i] = time.time() - start
            sizes[i] = getsize(s)
            for snirf in s:
                snirf.close()
            if VERBOSE:
                print('Loaded', len(TEST_FILES), 'SNIRF files of total size', sizes[i],
                      'in', str(times[i])[0:6], 'seconds with dynamic_loading =', mode)
        self.assertTrue(times[0] < times[1], msg='Dynamically-loaded files not loaded faster')
        self.assertTrue(sizes[0] < sizes[1], msg='Dynamically-loaded files not smaller in memory')
    
    
    def test_loading_saving(self):
        """
        Loads all files in filenames using Snirf in both dynamic and static mode, 
        saves them to a new file, compares the results using h5py and a naive cast.
        If returns True, all specified datasets are equivalent in the resaved files.
        """
        for i, mode in enumerate([True, False]):
            s1_paths = []
            s2_paths = []
            start = time.time()
            for file in TEST_FILES:
                snirf = Snirf(file, dynamic_loading=mode)
                s1_paths.append(file)
                new_path = file.split('.')[0] + '_unedited.snirf'
                snirf.save(new_path)
                s2_paths.append(new_path)
                snirf.close()
            if VERBOSE:
                print('Read and rewrote', len(TEST_FILES), 'SNIRF files in',
                str(time.time() - start)[0:6], 'seconds with dynamic_loading =', mode)
            
            for (fname1, fname2) in zip(s1_paths, s2_paths):
                dataset_equal_test(self, fname1, fname2)
    
    
    # def test_create_a_file(self):
    #     """
    #     Create a file from scratch using the data in the first test file. Compare
    #     the resulting file to a file created using the HDF library. 
    #     """
    #     for imode, mode in enumerate([True, False]):
    #         filename_in = TEST_FILES[0]
    #         filename_out = filename_in.split('.')[0] + '_' + str(imode)
    #         src_data = Snirf(filename_in, 'r+', dynamic_loading=mode)
    #         dst_data = Snirf(filename_out, 'w', dynamic_loading=mode)
            
    #         h5snirf = h5py.File(filename_out + '.hdf', 'w')
    #         h5snirf['nirs1/metaDataTags/SubjectID'] = src_data.nirs[0].metaDataTags.SubjectID 
    #         h5snirf['nirs1/metaDataTags/MeasurementDate'] = src_data.nirs[0].metaDataTags.MeasurementDate
    #         h5snirf['nirs1/metaDataTags/MeasurementTime'] = src_data.nirs[0].metaDataTags.MeasurementTime
    #         h5snirf['nirs1/metaDataTags/LengthUnit'] = src_data.nirs[0].metaDataTags.LengthUnit
    #         h5snirf['nirs1/metaDataTags/TimeUnit'] = src_data.nirs[0].metaDataTags.TimeUnit
            
    #         dst_data.nirs.appendGroup()
            
    #         # metaDataTags   
    #         dst_data.nirs[0].metaDataTags.SubjectID = src_data.nirs[0].metaDataTags.SubjectID
    #         dst_data.nirs[0].metaDataTags.MeasurementDate = src_data.nirs[0].metaDataTags.MeasurementDate
    #         dst_data.nirs[0].metaDataTags.MeasurementTime = src_data.nirs[0].metaDataTags.MeasurementTime
    #         dst_data.nirs[0].metaDataTags.LengthUnit = src_data.nirs[0].metaDataTags.LengthUnit
    #         dst_data.nirs[0].metaDataTags.TimeUnit = src_data.nirs[0].metaDataTags.TimeUnit
            
    #         # data
    #         h5snirf['nirs1/data1/time'] = src_data.nirs[0].data[0].time 
    #         h5snirf['nirs1/data1/dataTimeSeries'] = src_data.nirs[0].data[0].dataTimeSeries[:, 0]
            
    #         dst_data.nirs[0].data.appendGroup()
    #         dst_data.nirs[0].data[0].time = src_data.nirs[0].data[0].time 
    #         dst_data.nirs[0].data[0].dataTimeSeries = src_data.nirs[0].data[0].dataTimeSeries[:, 0]
            
    #         # measurementList
    #         h5snirf['nirs1/data1/measurementList1/sourceIndex'] = src_data.nirs[0].data[0].measurementList[0].sourceIndex
    #         h5snirf['nirs1/data1/measurementList1/detectorIndex'] = src_data.nirs[0].data[0].measurementList[0].detectorIndex
    #         h5snirf['nirs1/data1/measurementList1/wavelengthIndex'] = src_data.nirs[0].data[0].measurementList[0].wavelengthIndex
    #         h5snirf['nirs1/data1/measurementList1/dataType'] = src_data.nirs[0].data[0].measurementList[0].dataType
    #         h5snirf['nirs1/data1/measurementList1/dataTypeIndex'] = src_data.nirs[0].data[0].measurementList[0].dataTypeIndex
            
    #         dst_data.nirs[0].data[0].measurementList.appendGroup()
    #         dst_data.nirs[0].data[0].measurementList[0].sourceIndex = src_data.nirs[0].data[0].measurementList[0].sourceIndex
    #         dst_data.nirs[0].data[0].measurementList[0].detectorIndex = src_data.nirs[0].data[0].measurementList[0].detectorIndex
    #         dst_data.nirs[0].data[0].measurementList[0].wavelengthIndex = src_data.nirs[0].data[0].measurementList[0].wavelengthIndex
    #         dst_data.nirs[0].data[0].measurementList[0].dataType = src_data.nirs[0].data[0].measurementList[0].dataType
    #         dst_data.nirs[0].data[0].measurementList[0].dataTypeIndex = src_data.nirs[0].data[0].measurementList[0].dataTypeIndex
            
    #         # probe
    #         h5snirf['nirs1/probe/wavelengths'] = src_data.nirs[0].probe.wavelengths
    #         h5snirf['nirs1/probe/sourcePos2D'] = src_data.nirs[0].probe.sourcePos2D
    #         h5snirf['nirs1/probe/detectorPos2D'] = src_data.nirs[0].probe.detectorPos2D
            
    #         dst_data.nirs[0].probe.wavelengths = src_data.nirs[0].probe.wavelengths
    #         dst_data.nirs[0].probe.sourcePos2D = src_data.nirs[0].probe.sourcePos2D
    #         dst_data.nirs[0].probe.detectorPos2D = src_data.nirs[0].probe.detectorPos2D
            
    #         src_data.close()
    #         fname_dst = dst_data.filename
    #         fname_src = filename_in
    #         dst_data.save()
    #         dst_data.close()
    #         if VERBOSE:
    #             print(fname_dst, fname_src)
    #         h5snirf.close()
            
    #         dataset_equal_test(self, fname_dst, fname_src)
        

# -- Set up test working-directory --------------------------------------------

print('Deleting all files in', WD)
for file in os.listdir(WD):
    os.remove(WD + '\\' + file)

'Copying all files to WD'
for file in os.listdir(SNIRF_DIR):
    os.popen('copy ' + SNIRF_DIR + '\\' + file + ' ' + WD + '\\' + file)
time.sleep(1)  # Sleep while os exectues copy operation

TEST_FILES = [WD + '/' + file for file in os.listdir(WD)]


if __name__ == '__main__':
    result = unittest.main()
