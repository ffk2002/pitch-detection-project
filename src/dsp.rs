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
    pub fn collect_and_process(&mut self, samples: &[f32]) -> Option<f32> {
        self.buff.extend_from_slice(samples);

        //buffer is not yet full
        if self.buff.len()<self.window_size{
            return None;
        }else{
            //apply hann window to smooth edges of the window
            //create spectrum array of complex vals 
            let mut spectrum: Vec<Complex32> = std::iter::zip(&self.hann, &self.buff[..self.window_size])
                .map(|(&w, &s)| Complex32::new(w * s, 0.0))
                .collect();
            self.buff.drain(..self.window_size);
            let fft = self.planner.plan_fft_forward(self.window_size);
            fft.process(&mut spectrum);

            //apply suppressions to smooth out noise and pick highest energy bin
            // sample rate/window size = bin width
            //=48000/2048 = 23.4hz/bin
            let bin = self.suppressions(&spectrum)?+1;

            //calculate parabolic interpolation on discrete bin idxs
            //bins are 23.4Hz apart, and only return in discrete multiples
            //ex: input 110hz will return bin 5 (117.2Hz) as that is the closest
            // bin prediction
            //solutions: narrower bins or guess offset (parabolic interpolation)
            //source: cluade and indian guy on youtube
            let guess = self.peak_interpolation(bin, &spectrum);

            // the index of the bin containing the freq map
            Some((guess? as f32)*self.sample_rate/(self.window_size as f32))
        }
    }


    //used to suppress background noises with enough energy to dominate over notes
    pub fn suppressions(&mut self, spectrum: &Vec<Complex32>) -> Option<usize> {
        //find peak bin and mag
        let area = &spectrum[1..self.window_size/2];
        let mut peak_bin = 0usize;
        let mut peak_mag = 0.0f32;
        let mut mag_sum = 0.0f32;

        for (i, c) in area.iter().enumerate() {
            let mag = c.norm();
            mag_sum+=mag;
            if mag > peak_mag {
                peak_mag=mag;
                peak_bin=i;
            }

        }
        //then check to see if it passes noise floor test; energy > than floor*multiplier
        let avg_energy = mag_sum/area.len() as f32;
        if peak_mag < avg_energy*NOISE_FLOOR_MULT{
            return None
        }


        Some(peak_bin)
    }

    pub fn peak_interpolation(&mut self, bin: usize, spectrum: &Vec<Complex32>) -> Option<f32> {
        let mags = |i: usize| spectrum[i].norm();

        let a = mags(bin-1);
        let b = mags(bin);
        let c = mags(bin+1);
        let den = a-2.0*b+c;
        if den == 0.0{
            return Some(0.0);
        }
        let offset = 0.5*(a-c)/den;

        return Some(bin as f32+offset);
    }
}