# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 17:06:00 2022

@author: sstucker
"""

from pysnirf2 import Snirf
from tempfile import TemporaryFile
import h5py
from copy import deepcopy, copy


s1 = Snirf(r"C:\Users\sstucker\OneDrive\Documents\fnirs\Parya 12-3-2020\Parya - ninjaNIRS2020-12-3-12-09 Run 1 A.snirf")

s2 = Snirf(r"C:\Users\sstucker\OneDrive\Documents\fnirs\npc_training\Ex2_GLM_and_short_separation_regression\Ex2_GLM_and_short_separation_regression\SS_EXERCISE_1\Electrical_Stim_1.snirf")

dc = copy(s1)

s1.formatVersion = 2.0

s1.nirs[0].probe = s2.nirs[0].probe


x = h5py.File(TemporaryFile(), 'r+')
y = h5py.File(TemporaryFile(), 'r+')

x.create_group('/probe')

