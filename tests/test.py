import unittest
import pysnirf2
from pysnirf2 import Snirf, validateSnirf, loadSnirf, saveSnirf
import h5py
import os
import sys
import time
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping
import numpy as np
import shutil

VERBOSE = True  # Additional print statements in each test

# Need to run from the repository root
snirf_directory = os.path.join('tests', 'data')  # Sample data source
working_directory = os.path.join('tests', 'wd')  # Working directory for testing

if not os.path.isdir(working_directory):
    os.mkdir(working_directory)

if len(os.listdir(snirf_directory)) == 0:
    sys.exit('Failed to find test data in '+ snirf_directory)

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
    f1.close()
    f2.close()
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

class PySnirf2_Test(unittest.TestCase):
    
    def test_multidimensional_aux(self):
        """
        Test to ensure the validator permits multidimensional aux
        
        """
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    s.nirs[0].aux.appendGroup()
                    s.nirs[0].aux[-1].name = 'My2DAux'
                    s.nirs[0].aux[-1].time = np.linspace(0, 10, 100)
                    s.nirs[0].aux[-1].dataTimeSeries = np.random.random([100, 2])
                    if VERBOSE:
                        print("Created new aux channel:", s.nirs[0].aux[-1])
                    s.save()
                    if VERBOSE:
                        
                        s.validate().display()
                    self.assertTrue(s.validate(), msg="Incorrectly invalidated multidimensional aux signal!")
                self.assertTrue(validateSnirf(file), msg="Incorrectly invalidated multidimensional aux signal in file on disk!")
    
    def test_assignment(self):
        """
        Assign a Group and IndexedGroup element from one Snirf object to another. Validate that Datasets,
        misc metaDataTags and collections are copied successfully.
        
        """
        for i, mode in enumerate([False, True]):
            s2_paths = []
            start = time.time()
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                # Reassignment of same probe
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    same_probe = s.nirs[0].probe
                    self.assertTrue(isinstance(same_probe, pysnirf2.Probe), msg="Could not assign Probe reference")
                    same_probe.sourcePos3D = np.random.random([31, 3])
                    try:
                        s.nirs[0].probe = "foo"
                    except Exception as e:
                        self.assertTrue(type(e) is ValueError, msg="Faulty assignment to ProbeClass did not raise ValueError")
                    s.nirs[0].probe = same_probe
                # Assignment of new probe and measurement list element
                new_path = file.split('.')[0] + '_2.snirf'
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    s.save(new_path)
                    for mode2 in [False, True]:
                        print('Loading', new_path, 'with dynamic_loading=' + str(mode2))
                        with Snirf(new_path, 'r+', dynamic_loading=mode2) as s2:
                            # Group
                            new_srcpos3 = np.random.random([31, 3])
                            s2.nirs[0].probe.sourcePos3D = new_srcpos3
                            if VERBOSE:
                                print('Assigning probe from', new_path, 'to', file)
                            s.nirs[0].probe = s2.nirs[0].probe
                            s_location = str(s.nirs[0].probe._h.file.filename)
                            self.assertTrue(s_location == file, msg='Probe assignment unsuccessful, HDF5 file is ' + s_location + ', not ' + file)
                            self.assertTrue(np.all(s.nirs[0].probe.sourcePos3D == s2.nirs[0].probe.sourcePos3D), msg='Probe assignment unsuccessful: data not copied.')
                            # Indexed Group element
                            new_dataTypeLabel = str(int(np.random.random() * 10**5))[0:6]  # Random phony dataTypeLabel
                            s2.nirs[0].data[0].measurementList[0].dataTypeLabel = new_dataTypeLabel
                            if VERBOSE:
                                print('Assigning measurementList[0] from', new_path, 'to', file)
                            s.nirs[0].data[0].measurementList[0] = s2.nirs[0].data[0].measurementList[0]
                            s_location = str(s.nirs[0].data[0].measurementList[0]._h.file.filename)
                            self.assertTrue(s_location == file, msg='measurementList[0] assignment unsuccessful, HDF5 file is ' + s_location + ', not ' + file)
                            self.assertTrue(s.nirs[0].data[0].measurementList[0].dataTypeLabel == s2.nirs[0].data[0].measurementList[0].dataTypeLabel, msg='measurementList[0] assignment unsuccessful: data not copied.')
                # Assigning group and checking subgroups
                new_dataTypeLabel = str(int(np.random.random() * 10**5))[0:6]  # Random phony dataTypeLabel
                new_path = file.split('.')[0] + '_3.snirf'
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    s.save(new_path)
                    for mode2 in [False, True]:
                        print('Loading', new_path, 'with dynamic_loading=' + str(mode2))
                        with Snirf(new_path, 'r+', dynamic_loading=mode2) as s2:
                            # Edit measurementList, Assign Nirs Group element
                            s2.nirs[0].data[0].measurementList[0].dataTypeLabel = new_dataTypeLabel
                            # Remove a channel of data
                            del s2.nirs[0].data[0].measurementList[-1]
                            ch_count_old = np.shape(s2.nirs[0].data[0].dataTimeSeries)[1]
                            s2.nirs[0].data[0].dataTimeSeries = s2.nirs[0].data[0].dataTimeSeries[:, 0:-1]
                            ch_count_new = np.shape(s2.nirs[0].data[0].dataTimeSeries)[1]
                            # Add new metaDataTag
                            s2.nirs[0].metaDataTags.add('foo', 'bar')
                            if VERBOSE:
                                print("Assigned new dataTypeLabel to first measurementList element")
                                print("Reduced channel count from", ch_count_old, "to", ch_count_new)
                                print('Assigning nirs[0] from', new_path, 'to', file)
                            s.nirs[0] = s2.nirs[0]
                            new_new_path = new_path.split('.')[0] + '_resaved.snirf'
                            s.save(new_new_path)
                            # s_location = str(s.nirs[0].data[0].measurementList[0].filename)
                            # self.assertTrue(str(s.nirs[0].filename) == file, msg='nirs[0] assignment unsuccessful, HDF5 file for nirs element is ' + s_location + ', not ' + file)
                            # self.assertTrue(s_location == file, msg='measurementList[0] assignment unsuccessful, HDF5 file for measurementList element is ' + s_location + ', not ' + file)
                            self.assertTrue(len(s.nirs[0].data[0].measurementList) == len(s2.nirs[0].data[0].measurementList), msg='Assignment unsuccessful, IndexedGroup not successfully copied on assignment. Edited file had ' + str(len(s.nirs[0].data[0].measurementList)) + ' channels')
                            self.assertTrue(s.nirs[0].data[0].measurementList[0].dataTypeLabel == s2.nirs[0].data[0].measurementList[0].dataTypeLabel, msg='Assignment unsuccessful: data not copied.')
                            self.assertTrue(s.nirs[0].metaDataTags.foo == 'bar', msg='Assignment unsuccessful, failed to set the unspecified metaDataTag \'foo\'')
                # Resaved
                with Snirf(new_new_path, 'r+', dynamic_loading=mode) as s:
                    with Snirf(new_path, 'r+', dynamic_loading=mode2) as s2:
                            self.assertTrue(len(s.nirs[0].data[0].measurementList) == len(s2.nirs[0].data[0].measurementList) - 1, msg='Assignment unsuccessful after saving, IndexedGroup not successfully copied on assignment. Edited file had ' + str(len(s.nirs[0].data[0].measurementList)) + ' channels')
                            self.assertTrue(s.nirs[0].data[0].measurementList[0].dataTypeLabel == new_dataTypeLabel, msg='Assignment unsuccessful after saving: data not copied.')
                            self.assertTrue(s.nirs[0].metaDataTags.foo == 'bar', msg='Assignment unsuccessful after saving, failed to set the unspecified metaDataTag \'foo\'')
                            

    def test_copying(self):
        """
        Loads all files in filenames using Snirf in both dynamic and static mode, 
        saves copies to new files, compares the results using h5py and a naive cast.
        If returns True, all specified datasets are equivalent in the copied files.
        """
        for i, mode in enumerate([False, True]):
            s2_paths = []
            for file in self._test_files:
                new_path = file.split('.')[0] + '_copied.snirf'
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                    print('Making a copy of', file)
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    s2 = s.copy()
                    s.save()  # Save it, otherwise differences in IndexedGroup naming will raise issues with comparison
                    s2.save(new_path)
                    s2_paths.append(new_path)
            for (fname1, fname2) in zip(self._test_files, s2_paths):
                if VERBOSE:
                    print('Testing equality between', fname1, 'and', fname2)
                dataset_equal_test(self, fname1, fname2)
    
    
    def test_loading_saving_functions(self):
        """Test basic saving and loading interfaces `saveSnirf` and `loadSnirf`."""
        s1_paths = []
        s2_paths = []
        start = time.time()
        for file in self._test_files:
            s1 = loadSnirf(file)
            s1_paths.append(file)
            saveSnirf(file, s1)  # Otherwise, nirs/nirs1 inconsistencies will cause test to fail
            new_path = file.split('.')[0] + '_unedited.snirf'
            saveSnirf(new_path, s1)
            s2_paths.append(new_path)
            s1.close()
        for (fname1, fname2) in zip(s1_paths, s2_paths):
            dataset_equal_test(self, fname1, fname2)
    
    
    def test_disabled_logging(self):
        """Validate that no logs are created when logging is disabled."""
        for file in self._test_files:
            if VERBOSE:
                print('Loading', file)
            logfile = file.replace('.snirf', '.log')
            with Snirf(file, 'r', enable_logging=False) as s: 
                self.assertFalse(os.path.exists(logfile), msg='{} created even though enable_logging=False'.format(logfile))
    
    
    def test_enabled_logging(self):
        """Test log file creation."""
        for file in self._test_files:
            if VERBOSE:
                print('Loading', file)
            logfile = file.replace('.snirf', '.log')
            with Snirf(file, 'r', enable_logging=True) as s:
                self.assertTrue(os.path.exists(logfile), msg='{} was not created with enable_logging=True'.format(logfile))
                if VERBOSE:
                    print(logfile, 'contents:')
                    print('---------------------------------------------')
                    with open(logfile, 'r') as f:
                        [print(line) for line in f.readlines()]
                    print('---------------------------------------------')
    
    
    def test_unknown_coordsys_name(self):
        """Test that the validator warns about unknown coordinate system names if no description is present."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    if VERBOSE:    
                        print("Adding unrecognized coordinate system")
                    s.nirs[0].probe.coordinateSystem = 'MNIFoo27'
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=2)
                    self.assertTrue('UNRECOGNIZED_COORDINATE_SYSTEM' in [issue.name for issue in result.warnings], msg='Failed to raise warning about unknown coordinate system')
                    newname = file.split('.')[0] + '_coordinate_system_added'
                    s.save(newname)
                if VERBOSE:
                    print('Loading', newname, 'with dynamic_loading=' + str(mode))
                with Snirf(newname, 'r+', dynamic_loading=mode) as s:
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=2)
                    self.assertTrue('UNRECOGNIZED_COORDINATE_SYSTEM' in [issue.name for issue in result.warnings], msg='Failed to raise warning about unknown coordinate system in file saved to disk')
                    self.assertTrue(s.validate(), msg='File was incorrectly invalidated')


    def test_known_coordsys_name(self):
        """Test that the validator does NOT warn about known coordinate system names."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    if VERBOSE:    
                        print("Adding recognized coordinate system")
                    s.nirs[0].probe.coordinateSystem = 'MNIColin27'
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=2)
                    self.assertFalse('UNRECOGNIZED_COORDINATE_SYSTEM' in [issue.name for issue in result.warnings], msg='Failed to recognize known coordinate system')
                    newname = file.split('.')[0] + '_unknown_coordinate_system_added'
                    s.save(newname)
                if VERBOSE:
                    print('Loading', newname, 'with dynamic_loading=' + str(mode))
                with Snirf(newname, 'r+', dynamic_loading=mode) as s:
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=2)
                    self.assertFalse('UNRECOGNIZED_COORDINATE_SYSTEM' in [issue.name for issue in result.warnings], msg='Failed to recognize known coordinate system in file saved to disk')
                    self.assertTrue(s.validate(), msg='File was incorrectly invalidated')
    
    
    def test_unspecified_metadatatags(self):
        """Test that misc metaDataTags can be added, removed, saved and loaded."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    if VERBOSE:    
                        print("Adding metaDataTags 'foo', 'bar', and 'array_of_strings'")
                    s.save()  # Otherwise, nirs/nirs1 inconsistencies will cause test to fail
                    s.nirs[0].metaDataTags.add('foo', 'Hello')
                    s.nirs[0].metaDataTags.add('Bar', 'World')
                    s.nirs[0].metaDataTags.add('_array_of_strings', ['foo', 'bar'])
                    self.assertTrue(s.validate(), msg='adding the unspecified metaDataTags resulted in an INVALID file...')
                    self.assertTrue(s.nirs[0].metaDataTags.foo == 'Hello', msg='Failed to set the unspecified metadatatags')
                    self.assertTrue(s.nirs[0].metaDataTags.Bar == 'World', msg='Failed to set the unspecified metadatatags')
                    self.assertTrue(s.nirs[0].metaDataTags._array_of_strings[0] == 'foo', msg='Failed to set the unspecified metadatatags')
                    newname = file.split('.')[0] + '_unspecified_tags'
                    s.save(newname)
                    if VERBOSE:
                        print('Loading', newname, 'with dynamic_loading=' + str(mode))
                with Snirf(newname, 'r+', dynamic_loading=mode) as s:
                    self.assertTrue(s.nirs[0].metaDataTags.foo == 'Hello', msg='Failed to save the unspecified metadatatags to disk')
                    self.assertTrue(s.nirs[0].metaDataTags.Bar == 'World', msg='Failed to save the unspecified metadatatags to disk')
                    self.assertTrue(s.nirs[0].metaDataTags._array_of_strings[0] == 'foo', msg='Failed to save the unspecified metadatatags to disk')
                    s.nirs[0].metaDataTags.remove('foo')
                    s.nirs[0].metaDataTags.remove('Bar')
                    s.nirs[0].metaDataTags.remove('_array_of_strings')
                    s.save()
                dataset_equal_test(self, file, newname + '.snirf')
    
    
    def test_validator_required_probe_dataset_missing(self):
        """Test that the validator invalidates an a missing required dataset."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    probloc = s.nirs[0].probe.location
                    # Otherwise probloc will not find issue
                    if len(s.nirs) == 1:
                        probloc = probloc.replace('nirs1', 'nirs')
                    s.save()
                    del s.nirs[0].probe.sourcePos2D
                    del s.nirs[0].probe.detectorPos2D
                    if VERBOSE:
                        print('Deleted source and detector 2D positions from probe:')
                        print(s.nirs[0].probe)
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=3)
                    self.assertFalse(result[probloc + '/sourcePos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING not expected')
                    self.assertFalse(result[probloc + '/detectorPos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING not expected')
                    self.assertTrue(result[probloc + '/sourcePos2D'].name == 'OPTIONAL_DATASET_MISSING', msg='OPTIONAL_DATASET_MISSING expected')
                    self.assertTrue(result[probloc + '/detectorPos2D'].name == 'OPTIONAL_DATASET_MISSING', msg='OPTIONAL_DATASET_MISSING expected')
                    newname = file.split('.')[0] + '_optional_pos_missing'
                    newname2 = file.split('.')[0] + '_required_pos_missing'
                    s.save(newname)
                    del s.nirs[0].probe.sourcePos3D
                    del s.nirs[0].probe.detectorPos3D
                    s.save(newname2)
                result = validateSnirf(newname)
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result[probloc + '/sourcePos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING not expected')
                self.assertFalse(result[probloc + '/detectorPos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING not expected')
                self.assertTrue(result[probloc + '/sourcePos2D'].name == 'OPTIONAL_DATASET_MISSING', msg='OPTIONAL_DATASET_MISSING expected')
                self.assertTrue(result[probloc + '/detectorPos2D'].name == 'OPTIONAL_DATASET_MISSING', msg='OPTIONAL_DATASET_MISSING expected')
                result = validateSnirf(newname2)
                if VERBOSE:
                    print('Deleted source and detector 2D and 3D positions from probe:')
                    result.display(severity=3)
                self.assertTrue(result[probloc + '/sourcePos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING expected')
                self.assertTrue(result[probloc + '/detectorPos2D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING expected')
                self.assertTrue(result[probloc + '/sourcePos3D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING expected')
                self.assertTrue(result[probloc + '/detectorPos3D'].name == 'REQUIRED_DATASET_MISSING', msg='REQUIRED_DATASET_MISSING expected')
    
    
    def test_validator_required_group_missing(self):
        """Test that the validator invalidates an a missing required Group."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file, 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    del s.nirs[0].probe
                    if VERBOSE:
                        print('Performing local validation on probeless', s)
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=3)
                    self.assertFalse(result, msg='The Snirf object was incorrectly validated')
                    self.assertTrue('REQUIRED_GROUP_MISSING' in [issue.name for issue in result.errors], msg='REQUIRED_GROUP_MISSING not found')
                    newname = file.split('.')[0] + '_required_group_missing'
                    s.save(newname)

                if VERBOSE:
                    print('Performing file validation on probeless', newname + '.snirf')
                result = validateSnirf(newname)
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The file was incorrectly validated')
                self.assertTrue('REQUIRED_GROUP_MISSING' in [issue.name for issue in result.errors], msg='REQUIRED_GROUP_MISSING not found')
    
    
    def test_validator_required_dataset_missing(self):
        """Test that the validator invalidates an a missing required dataset."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files[0:1]:
                if VERBOSE:
                    print('Loading', file + '.snirf', 'with dynamic_loading=' + str(mode))
                with Snirf(file, 'r+', dynamic_loading=mode) as s:
                    del s.formatVersion
                    if VERBOSE:
                        print('Performing local validation on formatVersionless', s)
                    result = s.validate()
                    if VERBOSE:
                        result.display(severity=3)
                    self.assertFalse(result, msg='The Snirf object was incorrectly validated')
                    self.assertTrue('REQUIRED_DATASET_MISSING' in [issue.name for issue in result.errors], msg='REQUIRED_DATASET_MISSING not found')
                    newname = file.split('.')[0] + '_required_dataset_missing'
                    s.save(newname)

                if VERBOSE:
                    print('Performing file validation on formatVersionless', newname + '.snirf')
                result = validateSnirf(newname)
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The file was incorrectly validated')
                self.assertTrue('REQUIRED_DATASET_MISSING' in [issue.name for issue in result.errors], msg='REQUIRED_DATASET_MISSING not found')
    
    
    def test_validator_required_indexed_group_empty(self):
        """Test that the validator invalidates an empty indexed group."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files[0:1]:
                if VERBOSE:
                    print('Loading', file + '.snirf', 'with dynamic_loading=' + str(mode))
                s = Snirf(file, 'r+', dynamic_loading=mode)
                while len(s.nirs[0].data) > 0:
                    del s.nirs[0].data[0]
                if VERBOSE:
                    print('Performing local validation on dataless', s)
                result = s.validate()
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The Snirf object was incorrectly validated')
                self.assertTrue('REQUIRED_INDEXED_GROUP_EMPTY' in [issue.name for issue in result.errors], msg='REQUIRED_INDEXED_GROUP_EMPTY not found')
                newname = file.split('.')[0] + '_required_ig_empty'
                s.save(newname)
                s.close()
                if VERBOSE:
                    print('Performing file validation on dataless', newname + '.snirf')
                result = validateSnirf(newname)
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The file was incorrectly validated')
                self.assertTrue('REQUIRED_INDEXED_GROUP_EMPTY' in [issue.name for issue in result.errors], msg='REQUIRED_INDEXED_GROUP_EMPTY not found')
    
    
    def test_validator_invalid_measurement_list(self):
        """Test that the validator catches a measurementList which mismatches the dataTimeSeries in length."""
        for i, mode in enumerate([False, True]):
            for file in self._test_files[0:1]:
                if VERBOSE:
                    print('Loading', file + '.snirf', 'with dynamic_loading=' + str(mode))
                s = Snirf(file, 'r+', dynamic_loading=mode)
                s.nirs[0].data[0].measurementList.appendGroup()  # Add extra ml
                if VERBOSE:
                    print('Performing local validation on invalid ml', s)
                result = s.validate()
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The Snirf object was incorrectly validated')
                self.assertTrue('INVALID_MEASUREMENTLIST' in [issue.name for issue in result.errors], msg='INVALID_MEASUREMENTLIST not found')
                newname = file.split('.')[0] + '_invalid_ml'
                s.save(newname)
                s.close()
                if VERBOSE:
                    print('Performing file validation on invalid ml', newname + '.snirf')
                result = validateSnirf(newname)
                if VERBOSE:
                    result.display(severity=3)
                self.assertFalse(result, msg='The file was incorrectly validated')
                self.assertTrue('INVALID_MEASUREMENTLIST' in [issue.name for issue in result.errors], msg='INVALID_MEASUREMENTLIST not found')
    
    
    def test_edit_probe_group(self):
        """
        Edit some probe Group. Confirm they can be saved using save methods on
        the Snirf object and just the Group save method.
        """
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                if VERBOSE:
                    print('Loading', file + '.snirf', 'with dynamic_loading=' + str(mode))
                s = Snirf(file, 'r+', dynamic_loading=mode)
                
                group_save_file = file.split('.')[0] + '_edited_group_save.snirf'
                if VERBOSE:
                    print('Creating working copy for Group-level save', group_save_file)
                s.save(group_save_file)
                
                desired_probe_sourcelabels = ['S1_A', 'S2_A', 'S3_A', 'S4_A',
                                              'S5_A', 'S6_A', 'S7_A', 'S8_A',
                                              'S9_A', 'S10_A', 'S11_A', 'S12_A',
                                              'S13_A', 'S14_A', 'S15_A']
                desired_probe_uselocalindex = 1
                desired_probe_sourcepos3d = np.random.random([31, 3])
            
                s.nirs[0].probe.sourceLabels = desired_probe_sourcelabels
                s.nirs[0].probe.useLocalIndex = desired_probe_uselocalindex
                s.nirs[0].probe.sourcePos3D = desired_probe_sourcepos3d
                
                snirf_save_file = file.split('.')[0] + '_edited_snirf_save.snirf'
                print('Saving edited file to', snirf_save_file)
                s.save(snirf_save_file)
                
                print('Saving edited Probe group to', group_save_file)
                s.nirs[0].probe.save(group_save_file)
                
                s.close()
                
                for edited_filename in [snirf_save_file, group_save_file]:
                    
                    print('Loading', edited_filename, 'for comparison with dynamic_loading=' + str(mode))
                    s2 = Snirf(edited_filename, 'r+', dynamic_loading=mode)
                    
                    self.assertTrue((s2.nirs[0].probe.sourceLabels == desired_probe_sourcelabels).all(), msg='Failed to edit sourceLabels properly in ' + edited_filename) 
                    self.assertTrue(s2.nirs[0].probe.useLocalIndex == desired_probe_uselocalindex, msg='Failed to edit sourceLabels properly in ' + edited_filename) 
                    self.assertTrue((s2.nirs[0].probe.sourcePos3D == desired_probe_sourcepos3d).all(), msg='Failed to edit sourceLabels properly in ' + edited_filename) 
                    
                    s2.close()
                

    def test_add_remove_stim(self):
        """
        Use the interface to add and remove stim groups. Verify in memory and then in a reloaded file.
        """
        for i, mode in enumerate([False, True]):
            for file in self._test_files:
                file = self._test_files[0].split('.')[0]
                if VERBOSE:
                    print('Loading', file + '.snirf', 'with dynamic_loading=' + str(mode))
                s = Snirf(file, 'r+', dynamic_loading=mode)
                nstim = len(s.nirs[0].stim)
                s.nirs[0].stim.appendGroup()
                if VERBOSE:
                    print('Adding stim to', file + '.stim')
                self.assertTrue(len(s.nirs[0].stim) == nstim + 1, msg='IndexedGroup.appendGroup() failed')
                s.nirs[0].stim[-1].data = [[0, 10, 1], [5, 10, 1]]
                s.nirs[0].stim[-1].dataLabels = ['Onset', 'Duration', 'Amplitude']
                s.nirs[0].stim[-1].name = 'newCondition'
                newfile = file + '_added_stim_' + str(i)
                if VERBOSE:
                    print('Save As edited file to', newfile + '.stim')
                s.save(newfile)
                s.close()
                s2 = Snirf(newfile, 'r+', dynamic_loading=mode)
                self.assertTrue(len(s2.nirs[0].stim) == nstim + 1, msg='The new stim Group was not Saved As to ' + newfile + '.snirf')
                if VERBOSE:
                    print('Adding another stim group to', newfile + '.snirf and reloading it')
                s2.nirs[0].stim.appendGroup()
                s2.nirs[0].stim[-1].data = [[0, 10, 1], [5, 10, 1]]
                s2.nirs[0].stim[-1].dataLabels = ['Onset', 'Duration', 'Amplitude']
                s2.nirs[0].stim[-1].name = 'newCondition2'
                s2.save()
                s2.close()
                s3 = Snirf(newfile, 'r+', dynamic_loading=mode)
                self.assertTrue(len(s3.nirs[0].stim) == nstim + 2, msg='The new stim Group was not Saved to ' + newfile + '.snirf')
                if VERBOSE:
                    print('Removing all but one stim Group from', newfile + '.snirf and reloading it')
                name_to_keep = s3.nirs[0].stim[0].name
                while s3.nirs[0].stim[-1].name != name_to_keep:
                    if VERBOSE:
                        print('Deleting stim Group with name:', s3.nirs[0].stim[-1].name)
                    del s3.nirs[0].stim[-1]
                s3.close()
                s4 = Snirf(newfile, 'r+', dynamic_loading=mode)
                self.assertTrue(s4.nirs[0].stim[0].name == name_to_keep, msg='Failed to remove desired stim Groups from ' + newfile + '.snirf') 
                s4.close()
        
    
    def test_loading_saving(self):
        """
        Loads all files in filenames using Snirf in both dynamic and static mode, 
        saves them to a new file, compares the results using h5py and a naive cast.
        If returns True, all specified datasets are equivalent in the resaved files.
        """
        for i, mode in enumerate([False, True]):
            s1_paths = []
            s2_paths = []
            start = time.time()
            for file in self._test_files:
                snirf = Snirf(file, 'r+', dynamic_loading=mode)
                s1_paths.append(file)
                snirf.save()  # Otherwise, nirs/nirs1 will cause dataset_equal_test to fail
                new_path = file.split('.')[0] + '_unedited.snirf'
                snirf.save(new_path)
                s2_paths.append(new_path)
                snirf.close()
            if VERBOSE:
                print('Read and rewrote', len(self._test_files), 'SNIRF files in',
                str(time.time() - start)[0:6], 'seconds with dynamic_loading =', mode)
            
            for (fname1, fname2) in zip(s1_paths, s2_paths):
                dataset_equal_test(self, fname1, fname2)
    

    def test_dynamic(self):
        """
        Confirm that dynamically loaded files have smaller memory footprints
        and faster load-times than non dynamically loaded files
        """
        times = [-1, -1]
        sizes = [-1, -1]
        for i, mode in enumerate([False, True]):
            s = []
            start = time.time()
            for file in self._test_files:
                s.append(Snirf(file, 'r+', dynamic_loading=mode))
            times[i] = time.time() - start
            sizes[i] = getsize(s)
            for snirf in s:
                snirf.close()
            if VERBOSE:
                print('Loaded', len(self._test_files), 'SNIRF files of total size', sizes[i],
                      'in', str(times[i])[0:6], 'seconds with dynamic_loading =', mode)
        self.assertTrue(times[1] < times[0], msg='Dynamically-loaded files not loaded faster')
        self.assertTrue(sizes[1] < sizes[0], msg='Dynamically-loaded files not smaller in memory')
            
        
    def setUp(self):     
        if VERBOSE:
            print('Copying all test files to', working_directory)
        for file in os.listdir(snirf_directory):
            shutil.copy(os.path.join(snirf_directory, file),  os.path.join(working_directory, file))
            time.sleep(0.5)  # Sleep while executing copy operation
        
        self._test_files = [os.path.join(working_directory, file) for file in os.listdir(working_directory)]
        if len(self._test_files) == 0:
            sys.exit('Failed to set up test data working directory at '+ working_directory)
   
            
    def tearDown(self):
        if VERBOSE:
            print('Deleting all files in', working_directory)
        for file in os.listdir(working_directory):
            os.remove(os.path.join(working_directory, file))
            if VERBOSE:
                print('Deleted', os.path.join(working_directory, file))

# -- Set up test working-directory --------------------------------------------

if __name__ == '__main__':
    result = unittest.main()
