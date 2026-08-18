# pitch-detection-project

Real-time pitch detection from microphone input, written in Rust.

## What it does

Captures live audio from the default input device, runs it through an FFT
pipeline to estimate amplitude, dB level, and dominant frequency, and logs
each reading to `readings.csv`. A small Python script turns that CSV into a
self-contained HTML chart (`chart.html`) for visualizing captures.

## How it works

- **Capture thread** (`src/audio.rs`) reads audio frames from the mic via
  `cpal`, downmixes interleaved multi-channel samples to mono by averaging
  channels per frame, and pushes them into a lock-free ring buffer
  (`ringbuf`, 8192 `f32` samples).
- **Processing thread** (`src/audio.rs`) drains the ring buffer, computes
  RMS amplitude and dB, and feeds samples into an FFT-based pitch detector
  (`src/dsp.rs`). Each row (amplitude, dB, and frequency when available) is
  written to `readings.csv`.
- **Pitch detection** (`src/dsp.rs`) buffers samples into 2048-sample
  windows, applies a Hann window, runs an FFT (`rustfft`), and:
  - suppresses noise by discarding peaks that don't clear a
    noise-floor threshold (peak magnitude vs. average bin energy),
  - refines the peak bin with parabolic interpolation across the peak
    and its two neighboring bins, so the reported frequency isn't
    snapped to the nearest ~23 Hz-wide FFT bin.
- **Plotting** Python script to plot the recorded readings in the csv file 
  - run with `python3 scripts/plot_readings.py readings.csv -o chart.html`
  - used to plot dB, amplitude, and freq of recording


## Iterations
- **Hann window**
  - implemented a hann window on the buffered sample of audio, smoothing out 
    amplitude roughness around edges of samples when seen from changing pitches or 
    noises
- **Mono audio sampling**
  - detected frequencies returned from dsp module were consistently half of 
    inputted module
  - cause: when observing array of inputted samples of sound taken in from mic,
    noticed there were 2 copies of each sample, ex: [32 32 45 45 67 67 13 13]
    this is because the microphone reads in stereo, code needs to accomodate for it
    and was previously not
  - solution: took the average of each contiguous set of readings based on audio
    channel size. then took that one average and sent downstream
  - as a result, half as many samples were now sent, since num audio channels on
    this device=2, and counting over 48000Hz sample instead of 96000
- **Parabolic interpolation**
  - readings were now more accurate, within 10-15 Hz of actual recording. however,
    noticed the readings were off by at most a consistent margin. example, playing 
    sound of 110hz will record 117.2Hz. this is becuase bins are discrete counts
    and the frequency is calculated using bin*SAMPLE_RATE/bin_size. so if bin 
    size = 23.4 Hz, and max bin returned for 110hz is 5, freq returned is 5*23.4
    = ~117.2. freq returned will always be multiple of 23.4
  - solution: implemented parabolic interpolation, using 3 points to create 
    parabola that estimates max of the sample.


## Changes so far

1. **`add pitch detection`** — initial FFT-based pitch detection pipeline:
   ring buffer between capture and processing threads, CSV logging of
   amplitude/dB/frequency.
2. **`windowing and noise suppression`** — added a Hann window before the
   FFT to reduce spectral leakage, and a noise-floor check to avoid
   reporting a dominant frequency from background noise.
3. **`read mono audio from stereo, fix half freq read`** — fixed the
   capture thread to properly downmix stereo/multi-channel input to mono
   (previously misreading frequencies at roughly half their true value).
4. **`parabolic interpolation`** — replaced raw peak-bin frequency
   reporting with parabolic interpolation across the peak and its
   neighboring bins, reducing frequency error from a fraction of the
   ~23 Hz bin width down to a small fraction of a Hz for clean tones.


## TODO
1. MIDI note mapper
2. benchmarking
3. multiple peak detection
