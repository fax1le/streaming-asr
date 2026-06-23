import numpy as np
from numpy.typing import NDArray
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import time


def load_asr(card: str) -> ASRInferencePipeline:
    pipeline = ASRInferencePipeline(model_card=card)
    return pipeline


def transcribe(pipeline: ASRInferencePipeline, speech_segment: NDArray[np.float32]) -> tuple[list[str], float]:
    start = time.time()
    
    audio_input = [
        {
            "waveform": speech_segment,
            "sample_rate": 16000,
        }
    ]

    transcription = pipeline.transcribe(audio_input, lang=["kaz_Cyrl"], batch_size=1)
    latency = time.time() - start
    
    return transcription, latency
