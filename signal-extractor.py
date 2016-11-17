#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys, os, pyaudio, IPython
from scipy import signal as sig
from scipy.signal import resample
from scipy.io import wavfile


AUDIO_FS = 44100
SAMPLING_FS = 5000


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
def high_pass(signal, cutoff=20, nyq=9796):
	hp_filter = sig.firwin(11, cutoff, pass_zero=False, nyq=nyq)
	hp_filter /= np.sum(hp_filter)
	signal = sig.convolve(signal, hp_filter)
	# signal = sig.convolve(signal, hp_filter)
	return signal


"""
Plot and save frequency response as an image.
"""
def plot_fft(signal, directory_name):
	fft = np.fft.fft(signal)
	timestep = 1.0/SAMPLING_FS # number of seconds between samples/lines
	freq_bins = np.fft.fftfreq(len(fft), d=timestep)
	f = plt.figure()
	plt.plot(freq_bins, np.abs(fft))
	# f.suptitle('FFT of 44Hz Signal', fontsize=20) # put the title in the latex as a caption
	plt.xlabel('Frequency [Hz]', fontsize=16)
	plt.ylabel('Count', fontsize=16)
	# plt.ylim([0, 5000])
	plt.xlim([-500, 500])
	# plt.show()
	plt.savefig("data/plots/fft-" + directory_name + ".png")
	return


"""
Upsample to audio rate. Duration in seconds of the input (and output) signal
"""
def upsample(signal, duration):
	return resample(signal, int(AUDIO_FS * duration))


"""
Normalize signal to be between -1 to 1
"""
def normalize(signal):
	signal = signal - np.mean(signal)
	return signal / np.max(np.abs(signal))


"""
Threshold small frequencies in the signal to be 0, to reduce noise.
"""
def threshold(signal, epsilon):
	fft = np.fft.fft(signal)
	fft[np.where(np.abs(fft) < epsilon)] = 0
	return np.abs(np.fft.ifft(fft))


"""
Play a signal as audio.
"""
def play_audio(samples):
    p = pyaudio.PyAudio()
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=AUDIO_FS,output=True)
    ostream.write( samples.astype(np.float32).tostring() )
    p.terminate()
    return


"""
Save a signal as an audio WAV file.
"""
def save_audio(signal, directory_name):
	wavfile.write("data/reconstructions/" + directory_name+'.wav', AUDIO_FS, signal)
	return



"""
Main function for analyzing a video stream.
"""
def analyze_video(directory_name, should_plot=True, should_play=False, should_save=True, thresh=50):
	stream = get_frames("data/" + directory_name)
	signal = extract_signal(stream, pixel=512)
	signal = high_pass(signal, 20)

	signal = signal[20:-20] # there's some garbage in the beginning and end
	signal = normalize(signal)
	signal = threshold(signal, epsilon=thresh)

	if should_plot:
		plot_fft(signal, directory_name)
	
	signal = upsample(signal, 4.8)

	if should_play:
		play_audio(signal)
	
	if should_save:
		save_audio(signal, directory_name)
	# IPython.embed()
	return




if __name__ == "__main__":
    # analyze_video("video 1 frames")
    # analyze_video("video 2 frames")
    # analyze_video("video 3 frames")
    
    # analyze_video("44 Hz")
    # analyze_video("50 Hz")
    # analyze_video("50 Hz - 2")
    # analyze_video("100 Hz")
    # analyze_video("160 Hz")
    # analyze_video("250 Hz")
    # analyze_video("250 Hz - 2")
    # analyze_video("250 Hz - 3")
    # analyze_video("250 Hz - 4")
    # analyze_video("500 Hz")
    # analyze_video("500 Hz - 2")
    # analyze_video("100_160 Hz")
    # analyze_video("50_100 Hz")
    # analyze_video("50_100 Hz - 2")
    # analyze_video("50_160 Hz")

    # analyze_video("50hz_controlled")
    # analyze_video("160hz_controlled", thresh=90)
    # analyze_video("50_160hz_controlled")
    # analyze_video("50_160hz_seq_controlled")
    # analyze_video("skrillex")
    # analyze_video("rishi")


