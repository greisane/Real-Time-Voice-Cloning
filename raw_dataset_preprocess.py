from collections import deque
from pathlib import Path
from synthesizer.hparams import hparams
from utils.argutils import print_args
import argparse
import deepspeech
import librosa
import numpy as np
import re
import soundfile
import webrtcvad

def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, data, timestamp, duration):
        self.data = data
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(data, sample_rate, frame_duration):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(data):
        yield Frame(data[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(vad, frames, sample_rate, frame_duration, padding_duration):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration - The frame duration in milliseconds.
    padding_duration - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration / frame_duration)
    # We use a deque for our sliding window/ring buffer
    ring_buffer = deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the NOTTRIGGERED state
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.data, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in the ring buffer are
            # voiced frames, then enter the TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until we are NOTTRIGGERED,
                # but we have to start with the audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data and add it to the ring buffer
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are unvoiced,
            # then enter NOTTRIGGERED and yield whatever audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.data for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input, yield it
    if voiced_frames:
        yield b''.join([f.data for f in voiced_frames])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocesses and annotates audio files in a raw dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-f", "--file_pattern", type=str, default=r"Sample (?P<name>\d+)", help=\
        "Pattern that raw audio filenames must match. If available, capture group 'name' is used "
        "as the new filename.")
    parser.add_argument("-a", "--aggressiveness", type=int, default=3, help=\
        "Aggressiveness of silence detection for input audio chunking.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/<datasets_name>/train-clean/")
    parser.add_argument("--datasets_name", type=str, help=\
        "Name of the dataset directory to process.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--deepspeech_model", type=Path, default=Path("deepspeech-0.8.1-models.pbmm"),
        help="Path to the DeepSpeech model. Scorer file is also loaded if the filenames match.")
    parser.add_argument("--single_transcript", action="store_true", help=\
        "If True, generates a single transcript file in LibriSpeech fashion.", default=False)
    args = parser.parse_args()

    # Process the arguments
    assert args.deepspeech_model.exists()
    if not hasattr(args, 'out_dir'):
        args.out_dir = args.datasets_root / args.datasets_name / "train-clean"
    in_dirpath = args.datasets_root / args.datasets_name

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Prior setup
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)

    print(f"Loading DeepSpeech model from file {args.deepspeech_model.name}")
    ds = deepspeech.Model(str(args.deepspeech_model))
    deepspeech_scorer_path = args.deepspeech_model.with_suffix(".scorer")
    if deepspeech_scorer_path.exists():
        print(f"Loading DeepSpeech scorer from file {deepspeech_scorer_path.name}")
        ds.enableExternalScorer(str(deepspeech_scorer_path))
    # DeepSpeech demands a specific sample rate
    # Currently not handling this, which means only 16KHz is supported
    assert hparams.sample_rate == ds.sampleRate()

    vad = webrtcvad.Vad(mode=args.aggressiveness)
    if args.single_transcript:
        trans_fout = open((args.out_dir / args.out_dir.name).with_suffix(".trans.txt"), 'w')
        align_fout = open((args.out_dir / args.out_dir.name).with_suffix(".alignment.txt"), 'w')

    # Process all matching files
    file_num = 0
    for path in in_dirpath.rglob('*'):
        if not path.is_file():
            continue
        match = re.search(args.file_pattern, path.name)
        if not match:
            continue
        try:
            name = match.group('name')
        except IndexError:
            name = f"{file_num:04d}"
        file_num += 1

        print(f"Processing {path}")

        y, sample_rate = librosa.load(path, sr=None, mono=True)
        duration = librosa.get_duration(y, sr=sample_rate)
        if sample_rate != hparams.sample_rate:
            y = librosa.resample(y, sample_rate, hparams.sample_rate)
            sample_rate = hparams.sample_rate
        y = float2pcm(y)

        # Split audio by silences
        frames = list(frame_generator(y.tobytes(), sample_rate, frame_duration=30))
        segments = vad_collector(vad, frames, sample_rate, frame_duration=30, padding_duration=300)
        for segment_idx, segment_data in enumerate(segments):
            # Add some silence at the start, otherwise DeepSpeech will miss some words
            segment = np.frombuffer(segment_data, dtype=np.int16)
            segment = np.concatenate((np.zeros(8000, dtype=np.int16), segment))

            transcript = ds.sttWithMetadata(segment, 1).transcripts[0]

            # Save segment audio
            out_path = args.out_dir / f"{name}-{segment_idx:04d}"
            soundfile.write(str(out_path.with_suffix(".wav")), segment, hparams.sample_rate)

            # Save transcript
            word = ""
            words, timestamps = [], []
            for token_idx, token in enumerate(transcript.tokens):
                # Append character to word if it's not a space
                if not token.text.isspace():
                    if len(word) == 0:
                        # Log the start time of the new word
                        word_start_time = token.start_time
                    word = word + token.text
                # Word boundary is either a space or the last character in the array
                if token.text.isspace() or token_idx == len(transcript.tokens) - 1:
                    # words.append(word.upper())
                    words.append(word)
                    timestamps.append(f"{word_start_time:.3f}")
                    word, word_start_time = "", 0.0  # Reset

            if args.single_transcript:
                print(f'{out_path.name} {" ".join(words)}', file=trans_fout)
                print(f'{out_path.name} "{",".join(words)}" "{",".join(timestamps)}"', file=align_fout)
                print(f'{out_path.name}: {" ".join(words)}')
            else:
                with open(out_path.with_suffix(".txt"), 'w') as fout:
                    print(f'{" ".join(words)}.', file=fout)

    # Clean up
    if args.single_transcript:
        trans_fout.close()
        align_fout.close()