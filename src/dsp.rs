//fft processor file
// called from the processor thread, receives frame of samples of amplitudes
// package up the samples from the buffer into a full window, then send the window
// fft algorithm

// fft algo takes in list of samples and processes them and returns the dominant frequencies

use rustfft::{FftPlanner, num_complex::Complex32};

pub struct FftProcessor {
    hann: Vec<f32>,
    window_size: usize,
    sample_rate: f32,
    buff: Vec<f32>,
    hps: Vec<Complex32>,
    spectrum: Vec<Complex32>,
    planner: FftPlanner<f32>,
}

// const NOISE_FLOOR_MULT: f32 = 2.0;

impl FftProcessor {
    pub fn new(hann: Vec<f32>, sample_rate: f32, window_size: usize) -> Self {
        Self {
            hann,
            window_size,
            sample_rate,
            buff: vec![f32::default(); window_size],
            hps: vec![Complex32::default(); window_size / 2],
            spectrum: vec![Complex32::default(); window_size / 2],
            planner: FftPlanner::new(),
        }
    }

    //from the samples collected, populate buffer and send to fft planner when full
    //when fft returns magnitudes of the frequencies, use it to find the frequency with the highest
    // energy which will correspond to the pitch
    pub fn collect_and_process(&mut self, samples: &[f32]) -> Option<f32> {
        self.buff.extend_from_slice(samples);

        //buffer is not yet full
        if self.buff.len() < self.window_size {
            return None;
        } else {
            //apply hann window to smooth edges of the window
            //create spectrum array of complex vals
            let bin_hz = self.sample_rate / self.window_size as f32;
            self.spectrum = std::iter::zip(&self.hann, &self.buff[..self.window_size])
                .map(|(&w, &s)| Complex32::new(w * s, 0.0))
                .collect();

            //clear buffer after it is copied
            self.buff.drain(..self.window_size);
            let fft = self.planner.plan_fft_forward(self.window_size);
            fft.process(&mut self.spectrum);

            // implement hps in order to account for attenuations and higher multiple harmonics
            //DC rumble >0, 0 it out so it is not counted
            // let n = self.window_size / 2;
            let lo_bin = (20.0 / bin_hz).ceil() as usize;
            for i in 0..lo_bin {
                self.spectrum[i] = Complex32::default();
            }
            
            
            self.hps.copy_from_slice(&self.spectrum[..self.window_size/2]);
            for k in 2..=4{
                for i in 0..(self.window_size/2)/4{
                    self.hps[i]*=self.spectrum[i*k];
                }
            }

            //hps draws down multiplied frequencies to first 1/8 of vector
            //apply suppressions to smooth out noise and pick highest energy bin
            // sample rate/window size = bin width
            //=48000/2048 = 23.4hz/bin

            let top = self.window_size/8;
            let bin = self.suppressions(2, top).unwrap();

            // the index of the bin containing the freq map
            Some((bin as f32) * self.sample_rate / (self.window_size as f32))
        }
    }

    //used to suppress background noises with enough energy to dominate over notes
    pub fn suppressions(&mut self, bottom: usize, top: usize) -> Option<usize> {
        //find peak bin and mag
        let area: &mut [Complex32] = &mut self.hps[bottom..top];
        let mut peak_bin = 0usize;
        let mut peak_mag = 0.0f32;
        // let mut mag_sum = 0.0f32;

        for (i, c) in area.iter().enumerate() {
            let mag = c.norm();
            // mag_sum += mag;
            if mag > peak_mag {
                peak_mag = mag;
                peak_bin = i;
            }
        }
        //then check to see if it passes noise floor test; energy > than floor*multiplier
        // let avg_energy = mag_sum / area.len() as f32;
        // if peak_mag < avg_energy * NOISE_FLOOR_MULT {
        //     return None;
        // }

        // let len = 256;
        // let mid = 128;

        // let (&mut _, &mut median, &mut _) = area.select_nth_unstable_by(mid, |a, b| a.norm().total_cmp(&b.norm()));
        // if peak_mag < median.norm()*NOISE_FLOOR_MULT{
        //     return None;
        // }

        Some(peak_bin+bottom)
    }
}
