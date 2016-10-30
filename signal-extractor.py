#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys, os#, IPython
from scipy import signal as sig


"""
Opens the grayscale images of a directory and concatenates them together.
"""
def get_frames(directory_name):
	image_names = glob(str(directory_name) + "/*")
	stream = plt.imread(image_names[0])[:,:,0]  # grayscale, so arbitrarily take 1 channel
	for name in image_names[1:]:
		frame = plt.imread(name)[:,:,0]
		stream = np.concatenate((stream, frame), axis=0)
	return stream


"""
Extracts the signal of the i-th pixel from the video stream.
"""
def extract_signal(stream, pixel=0):
	return stream[:, pixel]


"""
High pass filter to remove frequencies below 20 hz.
"""
def high_pass(signal, cutoff=20):
	hp_filter = sig.firwin(11, cutoff, pass_zero=False, nyq=9796)
	hp_filter /= np.sum(hp_filter)
	return sig.convolve(signal, hp_filter)


# plot freq response
# upsample to audio rate
# play it


"""
Main function for analyzing a video stream.
"""
def analyze_video(i):
	stream = get_frames("data/video " + str(i) + " frames")
	signal = extract_signal(stream, pixel=512)
	filtered = high_pass(signal, 20)
	# IPython.embed()
	return filtered




if __name__ == "__main__":
    print analyze_video(1)
    # print analyze_video(2)

