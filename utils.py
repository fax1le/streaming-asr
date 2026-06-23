import numpy as np
import numpy.typing as npt
import soundfile as sf
import librosa

def convert_normalize_resample(pcm: bytes) -> npt.NDArray[np.float32]:
    samplesInt16 = np.frombuffer(pcm, dtype=np.int16)
    samplesFloat32 = samplesInt16.astype(np.float32) / 32768.0
    resampled_float32 = librosa.resample(samplesFloat32, orig_sr=48000, target_sr=16000)

    return resampled_float32


def write_wav(audio: npt.NDArray[np.float32]):
    try:
        sf.write("audio/input.wav", audio, 16000)
    except Exception:
        raise Exception("Failed to write wav file")

async def write_seg(audio: npt.NDArray[np.float32], path):
    try:
        sf.write(path, audio, 16000)
    except Exception:
        raise Exception("Failed to write wav file")    