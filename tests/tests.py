# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:12 2021

@author: sstucker
"""
import src.pysnirf2
from src.pysnirf2 import Snirf, NirsElement, StimElement
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
    the work of Aaron Hall https://stackoverflow.com/a/30316760
    Recursively iterate to sum size of object & members."""
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
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'): # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)

# TODO remove
def _print_keys(group):
    for key in group.keys():
        print(key)


FILENAME = 'subjA_run01.snirf'
SNIRF_DIR = 'testdata'  # Sample data source
WD = 'wd'  # Working directory for testing
print('Deleting all files in', WD)

for file in os.listdir(WD):
    os.remove(WD + '\\' + file)

path = WD + '\\' + FILENAME
os.popen('copy ' + SNIRF_DIR + '\\' + FILENAME + ' ' + path)
time.sleep(1)  # Sleep while os exectues copy operation

# %%

snirf = Snirf(path, dynamic_loading=False)
print('Loaded snirf from', path)
print(snirf)
print()
print('Adding some stim groups to stim...')
for i in range(5):
    snirf.nirs[0].stim.appendGroup()
print('Deleting one...')
del snirf.nirs[0].stim[3]
snirf.nirs[0].stim[3].name = 'assigned'
print(snirf.nirs[0].stim)
snirf.save('wd/two_new_stim.snirf')
snirf.close()

snirf2 = Snirf('wd/two_new_stim.snirf')
print('Loaded snirf from', 'wd/two_new_stim.snirf')
print(snirf2.nirs[0].stim)
for stim in snirf2.nirs[0].stim:
    print(stim)
snirf2.close()

# %%


class Test():
    
    def __init__(self):
        self._z = 'z'
        
        
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value
        
    @z.deleter
    def z(self):
        print('z deleted!')
        self._z = None

test = Test()
