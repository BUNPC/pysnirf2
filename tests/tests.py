# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:45:12 2021

@author: sstucker
"""
from pysnirf2_test import Snirf
import h5py

PATH = 'subjA_run01.snirf'

raw = h5py.File(PATH, 'r+')

snirf = Snirf(PATH)
