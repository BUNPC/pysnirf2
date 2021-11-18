# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:12 2021

@author: sstucker
"""
from pysnirf2 import Snirf
import h5py
import os
import sys
import time
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping

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


SNIRF_DIRECTORY = r'C:\Users\sstucker\OneDrive\Desktop\pysnirf2\tests\snirf'

s = Snirf(r"C:\Users\sstucker\OneDrive\Desktop\pysnirf2\tests\snirf\Electrical_Stim_2.snirf", dynamic_loading=True)

# %%

#snirfs = []
#
#for path in os.listdir(SNIRF_DIRECTORY):
##    for repeat in range(10):
#    # raw = h5py.File(PATH, 'r+')
#    start = time.time()
#    snirf = Snirf(SNIRF_DIRECTORY + '/' + path, dynamic_loading=True)
#    elapsed = time.time() - start
#    print('Loaded', path, 'size', getsize(snirf), 'bytes in', str(elapsed)[0:6], 'seconds')
#    snirfs.append(snirf)
#    print(len(snirf.nirs))
#    
## %%
#    
#s = snirfs[2]
#    
