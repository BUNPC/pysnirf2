# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:21:30 2021

@author: sstucker
"""

import os
import numpy as np
from pysnirf2 import *
import h5py

testfile = 'tests/wd/subjA_run01.snirf'

#s = Snirf(testfile)
s = Snirf(testfile, dynamic_loading=True)

z = s.nirs[0].probe.wavelengths
