from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import uvicorn
import numpy as np
import numpy.typing as npt
import torch

import silero
import utils
import omni

app = FastAPI()
vad_model = silero.load_vad()
asr_pipeline = omni.load_asr("omniASR_CTC_1B") #  swap other omni model: omniASR_LLM_1B

RATE = 16000
FRAME = 512


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    vad_iterator = silero.get_vad_iterator(vad_model, RATE)
    buffer: npt.NDArray[np.float32] = np.array([], dtype=np.float32)
    is_speech = False
    current_segment: list[npt.NDArray[np.float32]] = []

    seg_idx = 0 

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            buffer = np.append(buffer, utils.convert_normalize_resample(pcm_bytes))

            while len(buffer) >= FRAME:
                frame = buffer[:FRAME]
                buffer = buffer[FRAME:]

                segment_ready, is_speech = silero.decide_vad(
                    torch.from_numpy(frame), vad_iterator, is_speech
                )

                if is_speech:
                    current_segment.append(frame.copy())

                if segment_ready:  # means VAD detected "end"
                    segment_buffer = np.concatenate(current_segment)
                    current_segment = []

                    print(f"Segment: {segment_buffer.shape[0] / RATE:.2f}s")                    
                    # await utils.write_seg(segment_buffer, "audio/segment" + str(seg_idx) + ".wav")  used for testing
                        
                    transcription, latency = omni.transcribe(asr_pipeline, segment_buffer) # currently not async safe
                        
                    print(f"Transcription latency for Segment {seg_idx}: {latency * 1000:.0f} ms")
                        
                    await websocket.send_text(transcription[0])                    
                    seg_idx += 1

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"{datetime.now():%d-%m-%Y %H:%M:%S} connection was closed")
    finally:
        del buffer
        del current_segment

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8889)
