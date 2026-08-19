use crate::dsp;
use crate::midi;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{InputCallbackInfo, StreamConfig};
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::thread;
use std::time::Duration;

const RING_SIZE: usize = 8192;
const READINGS_CSV_PATH: &str = "readings.csv";
const FFT_WINDOW_SIZE: usize = 2048;

// create threads for capturing audio and processing signals
pub fn run(){
    //need to define device sample rate to provide to the processing thread, 
    // dont want to have to access cpal through processing thread
    let host        = cpal::default_host();
    let device      = host.default_input_device().expect("no input error");
    let sample_rate = device.default_input_config().expect("no default input config")
                        .sample_rate() as f32;
    println!("sample rate: {0}", sample_rate);

    // configure ring buffer
    let ring_buffer = HeapRb::<f32>::new(RING_SIZE);
    let (p, c)      = ring_buffer.split();

    //init threads on ring buffer parts
    let capture = thread::spawn(move || capture_thread(p));
    let process = thread::spawn(move || processer_thread(c, sample_rate));

    capture.join().unwrap();
    process.join().unwrap();
}

//simple thread to capture readings from the mic and send to audio and processing layers
//avoid allocation for real time audio capture reliability
fn capture_thread(mut producer: impl Producer<Item = f32> + Send + 'static){
    let host                               = cpal::default_host();
    let device                             = host.default_input_device().expect("no input found");
    let stream_config: StreamConfig        = device.default_input_config().expect("no default input config").into();
    let stream_channels: usize             = stream_config.channels as usize;

    let stream = device.build_input_stream(
        stream_config, 
        move |clip: &[f32], _: &InputCallbackInfo| { 
            producer.push_iter(
                //audio is captured in stereo format - [L0, R0, L1, R1, L2, R2]
                // average every pair of readings and send to processor thread 
                //solves halving of freq readings because buffer sent is 96000 samples
                //even though the bins are calculated with 48000 hz
                clip.chunks(stream_channels).map(|f| f.iter().sum::<f32>()/stream_channels as f32)
            );
        },
        |err| eprintln!("istream err: {err}"),
        None,
    ).expect("failed to build input stream");

    stream.play().expect("input stream start fail");


    loop{
        thread::park();
    }
}


fn processer_thread(mut consumer: impl Consumer<Item = f32>, sample_rate: f32){
    let window     = dsp::hann_window(FFT_WINDOW_SIZE);
    let mut fft    = dsp::FftProcessor::new(window, sample_rate, FFT_WINDOW_SIZE);
    let file       = File::create(READINGS_CSV_PATH).expect("failed to create readings csv");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "amplitude,db,freq_hz,notes,chord").expect("failed to write csv header");

    loop{
        let mut batch = Vec::new();

        while let Some(sample) = consumer.try_pop(){
            batch.push(sample);

        }

        if batch.len()>0{
            //rms formula - sqrt(sum(1-N)/N)
            let ss: f32 = batch.iter().map(|s| s*s).sum();
            let rms     = (ss/batch.len() as f32).sqrt();
            let db      = 20.0*rms.max(1e-12).log10();

            // only Some() once every FFT_WINDOW_SIZE samples have accumulated;
            // most rows leave these columns blank in the csv
            let freqs = fft.collect_and_process_multi(&batch);

            match freqs.as_deref() {
                Some(freqs) if !freqs.is_empty() => {
                    // strongest peak first = dominant frequency
                    let dominant = freqs[0];
                    // suppress overtones, map to midi, then match chord shape
                    let notes    = midi::freqs_to_notes(freqs);
                    let names    = notes.iter()
                                    .map(|&n| midi::midi_note_name(n))
                                    .collect::<Vec<_>>()
                                    .join(";"); //semicolons so the csv columns stay intact
                    let chord    = midi::identify_chord(&notes).unwrap_or_default();

                    if !chord.is_empty() {
                        println!("chord: {chord} ({names})");
                    }

                    writeln!(writer, "{rms:.4},{db:.4},{dominant:.2},{names},{chord}")
                }
                _ => writeln!(writer, "{rms:.4},{db:.4},,,"),
            }.expect("failed to write csv row");
            writer.flush().expect("failed to flush readings csv");
        }

        thread::sleep(Duration::from_millis(10));
    }
}