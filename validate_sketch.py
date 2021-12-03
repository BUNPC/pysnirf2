# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:21:30 2021

@author: sstucker
"""

import os
import numpy as np
from pysnirf2 import *
import h5py

testfile = 'tests/data/subjA_run01.snirf'

s = Snirf(testfile)
print(s)

# %%

valid, result = s.validate()



result.display(severity=2)