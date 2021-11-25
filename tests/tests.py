# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:12 2021

@author: sstucker
"""
import pysnirf2
from pysnirf2 import Snirf, NirsElement, StimElement
import h5py
import os
import sys
import time
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping
import numpy as np
#import property


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


def compare_snirf(snirf_path_1, snirf_path_2):
    '''
    Returns a struct pairing the Dataset names appearing in each SNIRF file with
    the result of comparing these values after a cast to numpy array. If None,
    the Dataset appears in only one of the two SNIRF files.
    '''
    f1 = h5py.File(snirf_path_1)
    f2 = h5py.File(snirf_path_2)
    loc_f1 = get_all_dataset_locations(f1)
    loc_f2 = get_all_dataset_locations(f2)
    
    results = {}
    
    locations = list(set(loc_f1 + loc_f2))
    for location in locations:
        if not (location in loc_f1 and location in loc_f2):
            results[location] = None
        else:
            eq = np.array(f1[location]) == np.array(f2[location])
            if not (type(eq) is np.bool_ or type(eq) is bool):
                eq = eq.all()   
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
        

def _print_keys(group):
    for key in group.keys():
        print(key)

def test_load(filename):
    snirf = Snirf(filename)
    snirf.close()
    print('Load PASSED')
    return True

def test_dynamic(filenames, verbose=True):
    times = [-1, -1]
    sizes = [-1, -1]
    for i, mode in enumerate([True, False]):
        s = []
        start = time.time()
        for file in filenames:
            s.append(Snirf(file, dynamic_loading=mode))
        times[i] = time.time() - start
        sizes[i] = getsize(s)
        for snirf in s:
            snirf.close()
        if verbose:
            print('Loaded', len(filenames), 'SNIRF files of total size', sizes[i],
                  'in', str(times[i])[0:6], 'seconds with dynamic_loading =', mode)
    if times[0] < times[1] and sizes[0] < sizes[1]:
        if verbose:
            print('Dynamic loading PASSED')
        return True
    else:
        if verbose:
            print('Dynamic loading FAILED')
        return False

def test_loading_saving(filenames, spec_locations, verbose=True, fn=None):
    """
    Loads all files in filenames using Snirf in both dynamic and static mode, 
    saves them to a new file, compares the results using h5py and a naive cast.
    If returns True, all specified datasets are equivalent in the resaved files.
    
    If callable modify_fn is passed, the SNIRF object is passed to it before
    resaving in order to test various editing features.
    """
    
    for i, mode in enumerate([True, False]):
        s1_paths = []
        s2_paths = []
        start = time.time()
        for file in filenames:
            snirf = Snirf(file, dynamic_loading=mode)
            s1_paths.append(file)
            if fn is not None:
                fn(snirf)  # Pass the snirf object to the function
                if verbose:
                    print('Applied', fn, 'to', file)
                new_path = file.split('.')[0] + '_edited.snirf'
            else:
                new_path = file.split('.')[0] + '_unedited.snirf'
            snirf.save(new_path)
            s2_paths.append(new_path)
            snirf.close()
        if verbose:
            print('Read and rewrote', len(filenames), 'SNIRF files in',
            str(time.time() - start)[0:6], 'seconds with dynamic_loading =', mode)
        
        for (fname1, fname2) in zip(s1_paths, s2_paths):
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
            if len(none_keys) > 0 and verbose:
                print('The following keys were not found in both SNIRF files:')
                print(none_keys)
            if len(false_keys) > 0 and verbose:
                print('The following keys were not equivalent:')
                print(false_keys)
#            if any([key in spec_locations and not ('metaDataTags' in key or 'stim0' in key or 'measurementList0' in key or 'data0' in key or 'aux0' in key or 'nirs0' in key) for key in none_keys]):
            if not any([('metaDataTags' in key or 'stim0' in key or 'measurementList0' in key or 'data0' in key or 'aux0' in key or 'nirs0' in key) for key in none_keys]):
                if verbose:
                    print('Specified dataset', key, 'not found in both SNIRF files! Test FAILED!')
                return False
            elif len(false_keys) > 0:
                if verbose:
                    print(len(false_keys), 'datasets not equivalent. Test FAILED!')
                return False
            else:
                if verbose:
                    print('All specified fields equivalent between', fname1, 'and', fname2)
    if verbose:
        print('All specified fields equivalent in all files. Test PASSED.')
    return True
                

def add_stim_condition(snirf):
    snirf.nirs[0].stim.appendGroup()
    snirf.nirs[0].stim[-1].name = 'appended'
    snirf.nirs[0].stim[-1].data = [0, 0, 0]


SNIRF_DIR = 'testdata'  # Sample data source
WD = 'wd'  # Working directory for testing

print('Deleting all files in', WD)
for file in os.listdir(WD):
    os.remove(WD + '\\' + file)

'Copying all files to WD'
for file in os.listdir(SNIRF_DIR):
    os.popen('copy ' + SNIRF_DIR + '\\' + file + ' ' + WD + '\\' + file)
    
time.sleep(1)  # Sleep while os exectues copy operation


# %% Tests

vb = True
test_files = [WD + '/' + file for file in os.listdir(WD)[0:1]]

print(test_files)

# %%

if test_dynamic(test_files, verbose=vb):
    print('Dynamic loading PASSED')
else:
    print('Dynamic loading FAILED')
    
# Read the specified locations in from file generated by gen
with open(r"C:\Users\sstucker\OneDrive\Documents\fnirs\pysnirf2\sstucker\pysnirf2\gen\locations.txt", 'r') as f:
    spec_locations = f.read().split('\n')

if test_loading_saving(test_files, spec_locations, verbose=vb):
    print('Load and resave without editing PASSED')
else:
    print('Load and resave without editing FAILED')
    
if test_loading_saving(test_files, spec_locations, verbose=vb, fn=add_stim_condition):
    print('Load and resave with added stim PASSED')
else:
    print('Load and resave with added stim FAILED')

# %% Save



# %% Edit



# %%

