# Audio-from-Speech
Reconstructing audio from a 1D high-speed video


In the data directory, extract the frames for each video:
```bash
mkdir video\ 1\ frames
ffmpeg -i raw\ videos/test_video1.avi -y -r 20.41 video\ 1\ frames/filename%03d.png
```

Then in the main directory, run the python script:
```bash
python signal-extractor.py
```

## Notes

### Old framerate:

* video duration: 2.45 seconds

* framerate: 20.41 fps

* number of images: 50

* image size: 480 x 1024


480 * 50 = 24000 lines scanned in 2.45 seconds ==> 9795.9 samples per second



### New framerate:

* video duration: 4.80 seconds

* framerate: 10.41 fps

* number of images: 50

* image size: 480 x 1024


480 * 50 = 24000 lines scanned in 4.8 seconds ==> 5000 samples per second

