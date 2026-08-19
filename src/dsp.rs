//fft processor file
// called from the processor thread, receives frame of samples of amplitudes
// package up the samples from the buffer into a full window, then send the window
// fft algorithm

// fft algo takes in list of samples and processes them and returns the dominant frequencies



use rustfft::{num_complex::Complex32, FftPlanner};

pub struct FftProcessor{
    hann        : Vec<f32>,
    window_size : usize,
    sample_rate : f32,
    buff        : Vec<f32>,
    planner     : FftPlanner<f32>,
}

const NOISE_FLOOR_MULT: f32 = 2.0;
//cap on how many peaks (potential simultaneous notes + overtones) to return per window
const MAX_PEAKS: usize = 6;
//peaks closer than this many bins to a stronger accepted peak are treated as
//sidelobes/leakage of the same note rather than a distinct pitch
const MIN_PEAK_SEPARATION_BINS: usize = 2;

//generates a Hann window of the given size for use with FftProcessor::new
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (size as f32 - 1.0)).cos())
        .collect()
}

impl FftProcessor{

    pub fn new(hann: Vec<f32>, sample_rate: f32, window_size: usize) -> Self{
        Self{
            hann,
            window_size,
            sample_rate,
            buff: Vec::with_capacity(window_size),
            planner: FftPlanner::new(),
        }
    }

    //from the samples collected, populate buffer and send to fft planner when full
    //when fft returns magnitudes of the frequencies, use it to find the frequency with the highest
    // energy which will correspond to the pitch
    pub fn _collect_and_process(&mut self, samples: &[f32]) -> Option<f32> {
        //the dominant frequency is the strongest peak, which multi returns first
        self.collect_and_process_multi(samples)
            .and_then(|freqs| freqs.first().copied())
    }

    //same as collect_and_process, but returns every spectral peak passing the
    //noise floor, strongest first - needed for chord detection where several
    //notes sound at once
    //returns None while the window is still filling, Some(vec) once processed
    // (empty vec when nothing passed the noise floor)
    pub fn collect_and_process_multi(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.buff.extend_from_slice(samples);

        //buffer is not yet full
        if self.buff.len()<self.window_size{
            return None;
        }

        //apply hann window to smooth edges of the window
        //create spectrum array of complex vals
        let mut spectrum: Vec<Complex32> = std::iter::zip(&self.hann, &self.buff[..self.window_size])
            .map(|(&w, &s)| Complex32::new(w * s, 0.0))
            .collect();
        self.buff.drain(..self.window_size);
        let fft = self.planner.plan_fft_forward(self.window_size);
        fft.process(&mut spectrum);

        //find local maxima passing the noise floor, strongest first
        // sample rate/window size = bin width
        //=48000/2048 = 23.4hz/bin
        let peaks = self.find_peaks(&spectrum);

        //calculate parabolic interpolation on discrete bin idxs
        //bins are 23.4Hz apart, and only return in discrete multiples
        //ex: input 110hz will return bin 5 (117.2Hz) as that is the closest
        // bin prediction
        //solutions: narrower bins or guess offset (parabolic interpolation)
        //source: cluade and indian guy on youtube
        let freqs = peaks.iter()
            .filter_map(|&bin| self.peak_interpolation(bin, &spectrum))
            .map(|guess| guess*self.sample_rate/(self.window_size as f32))
            .collect();

        Some(freqs)
    }


    //finds local maxs in spectrum that pass the noise floor test,
    //returned strongest first; peaks closer than MIN_PEAK_SEPARATION_BINS to an
    //already accepted (stronger) peak are skipped as sidelobes of the same note
    fn find_peaks(&self, spectrum: &[Complex32]) -> Vec<usize> {
        let half = self.window_size/2;

        //avg energy of the analysis area for the noise floor test;

        let mag_sum: f32 = spectrum[1..half].iter().map(|c| c.norm()).sum();
        let avg_energy   = mag_sum/(half-1) as f32;
        let floor        = (avg_energy*NOISE_FLOOR_MULT).max(1e-9);

        //local maxima above the noise floor; bins 2..half-1 for range in 
        // parabolic interp fn
        let mut candidates: Vec<(usize, f32)> = (2..half-1)
            .filter_map(|i| {
                let mag = spectrum[i].norm();
                let is_peak = mag > spectrum[i-1].norm() && mag >= spectrum[i+1].norm();
                (is_peak && mag > floor).then_some((i, mag))
            })
            .collect();

        //strongest first, then greedily accept peaks far enough from accepted ones
        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut peaks: Vec<usize> = Vec::new();
        for (bin, _mag) in candidates {
            if peaks.len() >= MAX_PEAKS {
                break;
            }
            if peaks.iter().all(|&p| bin.abs_diff(p) >= MIN_PEAK_SEPARATION_BINS) {
                peaks.push(bin);
            }
        }

        peaks
    }

    pub fn peak_interpolation(&self, bin: usize, spectrum: &[Complex32]) -> Option<f32> {
        //interpolate over log-magnitude rather than raw magnitude - a Hann window's
        //mainlobe is close to parabolic on a log scale, but not on a linear one, so
        //fitting the parabola directly to magnitude leaves a systematic bias
        //(observed up to ~1.2Hz on synthetic tones vs ~0.3-0.4Hz with log-magnitude)
        let mags = |i: usize| (spectrum[i].norm() + 1e-12).ln();

        let a = mags(bin-1);
        let b = mags(bin);
        let c = mags(bin+1);
        let den = a-2.0*b+c;
        if den == 0.0{
            return Some(bin as f32);
        }
        let offset = 0.5*(a-c)/den;

        return Some(bin as f32+offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32  = 48000.0;
    const WINDOW_SIZE: usize = 2048;
    // bin width = SAMPLE_RATE / WINDOW_SIZE ≈ 23.4 Hz; allow a small
    // fraction of a Hz of slack for interpolation/floating point error
    const TOLERANCE_HZ: f32 = 1.0;

    //generates `num_samples` of a pure sine tone at `freq_hz`, sampled at SAMPLE_RATE
    fn sine_wave(freq_hz: f32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SAMPLE_RATE).sin())
            .collect()
    }

    fn processor() -> FftProcessor {
        let window: Vec<f32> = (0..WINDOW_SIZE).map(|i| 0.5 - 0.5*(2.0 * std::f32::consts::PI*i as f32/(WINDOW_SIZE as f32-1.0)).cos()).collect();
        FftProcessor::new(window, SAMPLE_RATE, WINDOW_SIZE)
    }

    //feeds one window's worth of a pure tone and returns the detected frequency
    fn detect(freq_hz: f32) -> Option<f32> {
        let mut fft = processor();
        let samples = sine_wave(freq_hz, WINDOW_SIZE);
        fft.collect_and_process(&samples)
    }

    #[test]
    fn detects_a4_440hz() {
        let detected = detect(440.0).expect("expected a frequency reading");
        assert!(
            (detected - 440.0).abs() < TOLERANCE_HZ,
            "expected ~440 Hz, got {detected}"
        );
    }

    #[test]
    fn detects_common_piano_notes() {
        // (note name, frequency in Hz)
        let notes = [
            ("C4", 261.63),
            ("E4", 329.63),
            ("A4", 440.00),
            ("C5", 523.25),
            ("A5", 880.00),
        ];

        for (name, freq) in notes {
            let detected = detect(freq).expect("expected a frequency reading");
            assert!(
                (detected - freq).abs() < TOLERANCE_HZ,
                "{name}: expected ~{freq} Hz, got {detected}"
            );
        }
    }

    #[test]
    fn silence_returns_none() {
        let mut fft = processor();
        let samples = vec![0.0f32; WINDOW_SIZE];
        assert_eq!(fft.collect_and_process(&samples), None);
    }

    #[test]
    fn partial_window_returns_none() {
        let mut fft = processor();
        let samples = sine_wave(440.0, WINDOW_SIZE / 2);
        assert_eq!(fft.collect_and_process(&samples), None);
    }

    #[test]
    fn buffers_samples_across_multiple_calls() {
        let mut fft = processor();
        let samples = sine_wave(440.0, WINDOW_SIZE);

        // feed the window in two chunks; only the second call should
        // have enough buffered samples to produce a reading
        assert_eq!(fft.collect_and_process(&samples[..WINDOW_SIZE / 2]), None);
        let detected = fft
            .collect_and_process(&samples[WINDOW_SIZE / 2..])
            .expect("expected a frequency reading once window is full");
        assert!(
            (detected - 440.0).abs() < TOLERANCE_HZ,
            "expected ~440 Hz, got {detected}"
        );
    }

    #[test]
    fn detects_c_major_chord() {
        // C4 + E4 + G4 played simultaneously; multi should surface all three
        // peaks, which map to midi notes 60/64/67 and identify as Cmaj
        let mut fft = processor();
        let samples: Vec<f32> = (0..WINDOW_SIZE)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                let two_pi = 2.0 * std::f32::consts::PI;
                ((two_pi * 261.63 * t).sin()
                    + (two_pi * 329.63 * t).sin()
                    + (two_pi * 392.0 * t).sin())
                    / 3.0
            })
            .collect();

        let freqs = fft
            .collect_and_process_multi(&samples)
            .expect("expected a processed window");
        let notes = crate::midi::freqs_to_notes(&freqs);
        assert_eq!(notes, vec![60, 64, 67], "freqs were {freqs:?}");
        assert_eq!(
            crate::midi::identify_chord(&notes).as_deref(),
            Some("Cmaj")
        );
    }

    #[test]
    fn low_amplitude_tone_still_detected() {
        // the noise floor check is a peak/average energy ratio, not an
        // absolute amplitude threshold, so a quiet but clean tone should
        // still be detected correctly
        let mut fft = processor();
        let samples: Vec<f32> = sine_wave(440.0, WINDOW_SIZE)
            .iter()
            .map(|s| s * 0.0001)
            .collect();
        let detected = fft
            .collect_and_process(&samples)
            .expect("expected a frequency reading despite low amplitude");
        assert!(
            (detected - 440.0).abs() < TOLERANCE_HZ,
            "expected ~440 Hz, got {detected}"
        );
    }
}