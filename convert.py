#!/usr/bin/env python

"""
Converts all avi videos in "data/raw videos/" directory to frames
"""

from subprocess import call
from glob import glob

a = glob('data/raw videos/*')
a = [b[16:-4] for b in a]

for b in a:
    call(['mkdir', 'data/'+b])
    call(['ffmpeg', '-i', 'data/raw videos/'+b+'.avi', '-y', '-r', '20.41', 'data/'+b+'/frame%03d.png'])
