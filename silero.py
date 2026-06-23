from typing import Any
import torch

from silero_vad import load_silero_vad, VADIterator


def load_vad() -> Any:
    model = load_silero_vad()
    return model


def get_vad_iterator(model, rate: int) -> VADIterator:
    return VADIterator(
        model,
        sampling_rate=rate,
        threshold=0.5,
        min_silence_duration_ms=350,
        speech_pad_ms=100,
    )


def decide_vad(
    frame_tensor: torch.Tensor,
    vad_iterator: VADIterator,
    is_speech: bool,
) -> tuple[bool, bool]:

    speech_event = vad_iterator(frame_tensor)

    if speech_event and "start" in speech_event:
        is_speech = True

    segment_ready = False
    if speech_event and "end" in speech_event:
        is_speech = False
        segment_ready = True

    return segment_ready, is_speech
