#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys, os, IPython
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


"""
Plot and save frequency response as an image.
"""
def plot_fft(signal, directory_name):
	fft = np.fft.fft(signal)
	timestep = 1.0/5000 # number of seconds between samples/lines
	freq_bins = np.fft.fftfreq(len(fft), d=timestep)
	f = plt.figure()
	plt.plot(freq_bins, np.abs(fft))
	# f.suptitle('FFT of 44Hz Signal', fontsize=20) # put the title in the latex as a caption
	plt.xlabel('Frequency [Hz]', fontsize=16)
	plt.ylabel('Count', fontsize=16)
	plt.ylim([0, 5000])
	plt.xlim([-500, 500])
	# plt.show()
	plt.savefig("data/plots/fft-" + directory_name + ".png")
	return

# upsample to audio rate
# play it


"""
Main function for analyzing a video stream.
"""
def analyze_video(directory_name):
	stream = get_frames("data/" + directory_name)
	signal = extract_signal(stream, pixel=512)
	# filtered = high_pass(signal, 20)
	plot_fft(signal, directory_name)

	# IPython.embed()
	return




if __name__ == "__main__":
    # analyze_video("video 1 frames")
    # analyze_video("video 2 frames")
    # analyze_video("video 3 frames")
    
    analyze_video("44 Hz")
    analyze_video("50 Hz")
    analyze_video("50 Hz - 2")
    analyze_video("100 Hz")
    analyze_video("160 Hz")
    analyze_video("250 Hz")
    analyze_video("250 Hz - 2")
    analyze_video("250 Hz - 3")
    analyze_video("250 Hz - 4")
    analyze_video("500 Hz")
    analyze_video("500 Hz - 2")
    analyze_video("100_160 Hz")
    analyze_video("50_100 Hz")
    analyze_video("50_100 Hz - 2")
    analyze_video("50_160 Hz")


