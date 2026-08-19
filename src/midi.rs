//midi mapping layer
//converts frequencies from the fft processor into midi note numbers,
//note names, and identifies the chord being played from the set of
//simultaneous pitch classes

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

//tolerance for treating a peak as an integer harmonic of a stronger, lower peak
const HARMONIC_TOLERANCE: f32 = 0.035;

//chord templates as intervals (semitones) above the root pitch class
//matched as exact sets, 4-note templates listed alongside triads since
//set size must match anyway
const CHORD_TEMPLATES: &[(&str, &[u8])] = &[
    ("maj",  &[0, 4, 7]),
    ("min",  &[0, 3, 7]),
    ("dim",  &[0, 3, 6]),
    ("aug",  &[0, 4, 8]),
    ("sus2", &[0, 2, 7]),
    ("sus4", &[0, 5, 7]),
    ("maj7", &[0, 4, 7, 11]),
    ("7",    &[0, 4, 7, 10]),
    ("min7", &[0, 3, 7, 10]),
    ("dim7", &[0, 3, 6, 9]),
];

pub fn freq_to_midi(freq: f32) -> Option<u8> {
    if freq <= 0.0 {
        return None;
    }
    let note = 69.0 + 12.0 * (freq / 440.0).log2();
    let rounded = note.round();
    if (0.0..=127.0).contains(&rounded) {
        Some(rounded as u8)
    } else {
        None
    }
}

pub fn midi_note_name(midi: u8) -> String {
    let octave = (midi / 12) as i32 - 1;
    format!("{}{}", NOTE_NAMES[(midi % 12) as usize], octave)
}

//maps fft peaks (ordered strongest-first) to deduped midi notes
//a peak sitting at an integer multiple of a stronger lower peak is likely an
//overtone of that note rather than a separately played key, so it is dropped -
//without this a single piano note's harmonics would read as a fake chord
//(e.g. the 5th harmonic adds a major third that can mislabel minor chords)
pub fn freqs_to_notes(freqs: &[f32]) -> Vec<u8> {
    let mut kept: Vec<f32> = Vec::new();

    for &f in freqs {
        let is_harmonic = kept.iter().any(|&k| {
            if f <= k {
                return false;
            }
            let ratio = f / k;
            let nearest = ratio.round();
            nearest >= 2.0 && (ratio - nearest).abs() / nearest < HARMONIC_TOLERANCE
        });
        if !is_harmonic {
            kept.push(f);
        }
    }

    let mut notes: Vec<u8> = kept.iter().filter_map(|&f| freq_to_midi(f)).collect();
    notes.sort_unstable();
    notes.dedup();
    notes
}

//identify the chord from a set of midi notes by reducing to pitch classes
//and matching interval sets against the templates
//tries the bass note as root first so inversions still name the right chord
pub fn identify_chord(notes: &[u8]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }

    let mut pitch_classes: Vec<u8> = notes.iter().map(|n| n % 12).collect();
    pitch_classes.sort_unstable();
    pitch_classes.dedup();

    //wait fr distinct pitch classes
    if pitch_classes.len() < 3 {
        return None;
    }

    //root candidates: bass note's pitch class first, then the rest
    let bass_pc = notes.iter().min().unwrap() % 12;
    let mut candidates = vec![bass_pc];
    candidates.extend(pitch_classes.iter().copied().filter(|&pc| pc != bass_pc));

    for root in candidates {
        let mut intervals: Vec<u8> = pitch_classes
            .iter()
            .map(|&pc| (pc + 12 - root) % 12)
            .collect();
        intervals.sort_unstable();

        for (name, template) in CHORD_TEMPLATES {
            if intervals == *template {
                return Some(format!("{}{}", NOTE_NAMES[root as usize], name));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_maps_to_midi() {
        assert_eq!(freq_to_midi(440.0), Some(69)); // A4
        assert_eq!(freq_to_midi(261.63), Some(60)); // C4
        assert_eq!(freq_to_midi(27.5), Some(21)); // A0, lowest piano key
        assert_eq!(freq_to_midi(0.0), None);
        assert_eq!(freq_to_midi(-5.0), None);
        assert_eq!(freq_to_midi(30000.0), None); // beyond midi range
    }

    #[test]
    fn midi_note_names() {
        assert_eq!(midi_note_name(60), "C4");
        assert_eq!(midi_note_name(69), "A4");
        assert_eq!(midi_note_name(61), "C#4");
        assert_eq!(midi_note_name(21), "A0");
    }

    #[test]
    fn identifies_major_and_minor_triads() {
        assert_eq!(identify_chord(&[60, 64, 67]).as_deref(), Some("Cmaj")); // C E G
        assert_eq!(identify_chord(&[57, 60, 64]).as_deref(), Some("Amin")); // A C E
        assert_eq!(identify_chord(&[62, 66, 69]).as_deref(), Some("Dmaj")); // D F# A
    }

    #[test]
    fn identifies_inversions() {
        // first inversion C major: E G C
        assert_eq!(identify_chord(&[64, 67, 72]).as_deref(), Some("Cmaj"));
        // second inversion A minor: E A C
        assert_eq!(identify_chord(&[64, 69, 72]).as_deref(), Some("Amin"));
    }

    #[test]
    fn identifies_seventh_chords() {
        assert_eq!(identify_chord(&[60, 64, 67, 70]).as_deref(), Some("C7"));
        assert_eq!(identify_chord(&[60, 64, 67, 71]).as_deref(), Some("Cmaj7"));
        assert_eq!(identify_chord(&[57, 60, 64, 67]).as_deref(), Some("Amin7"));
    }

    #[test]
    fn too_few_notes_is_not_a_chord() {
        assert_eq!(identify_chord(&[]), None);
        assert_eq!(identify_chord(&[60]), None);
        assert_eq!(identify_chord(&[60, 67]), None); // dyad
        assert_eq!(identify_chord(&[60, 72]), None); // octave, one pitch class
    }

    #[test]
    fn harmonics_are_suppressed() {
        // A3 at 220hz with its 2nd/3rd/4th harmonics, strongest-first:
        // only the fundamental should survive as a note
        let notes = freqs_to_notes(&[220.0, 440.0, 660.0, 880.0]);
        assert_eq!(notes, vec![57]); // A3

        // a real triad has non-integer ratios and survives intact
        let notes = freqs_to_notes(&[261.63, 329.63, 392.0]); // C4 E4 G4
        assert_eq!(notes, vec![60, 64, 67]);
    }
}
