"""OpenAI-compatible TTS server backed by Kokoro (PyTorch GPU)."""

import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts-server")

SAMPLE_RATE = 24000

# OpenAI voice name -> Kokoro voice name
VOICE_MAP = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "shimmer": "af_sky",
}

# Language code prefix -> KPipeline lang_code
LANG_PREFIXES = {
    "a": "a",  # American English
    "b": "b",  # British English
    "e": "e",  # Spanish
    "f": "f",  # French
    "h": "h",  # Hindi
    "i": "i",  # Italian
    "j": "j",  # Japanese
    "p": "p",  # Portuguese
    "z": "z",  # Mandarin Chinese
}

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}

# One pipeline per language code, lazily initialized
pipelines: dict[str, KPipeline] = {}
whisper_model: WhisperModel | None = None


def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in pipelines:
        logger.info(f"Loading pipeline for lang_code='{lang_code}'...")
        pipelines[lang_code] = KPipeline(lang_code=lang_code)
        logger.info(f"Pipeline '{lang_code}' ready.")
    return pipelines[lang_code]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model
    # Pre-load American English pipeline (most common)
    logger.info("Loading Kokoro model (PyTorch GPU)...")
    get_pipeline("a")
    logger.info("Loading Whisper large-v3 (faster-whisper, GPU)...")
    whisper_model = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
    logger.info("Whisper model ready.")
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Kokoro TTS", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str = Field(..., max_length=4096)
    voice: str = "alloy"
    response_format: Literal["mp3", "wav", "flac", "opus", "pcm", "aac"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)


def encode_audio(samples: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "pcm":
        # 16-bit little-endian PCM, same as OpenAI
        pcm = (samples * 32767).astype(np.int16)
        buf.write(pcm.tobytes())
    elif fmt == "opus":
        sf.write(buf, samples, SAMPLE_RATE, format="OGG", subtype="VORBIS")
    elif fmt == "mp3":
        sf.write(buf, samples, SAMPLE_RATE, format="MP3")
    elif fmt == "aac":
        # soundfile doesn't support AAC; fall back to WAV
        sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    else:
        sf.write(buf, samples, SAMPLE_RATE, format=fmt.upper())
    return buf.getvalue()


def resolve_voice(name: str) -> str:
    """Map an OpenAI voice name or pass through a Kokoro voice name."""
    if name in VOICE_MAP:
        return VOICE_MAP[name]
    # Accept any voice name with a valid language prefix
    if len(name) >= 3 and name[0] in LANG_PREFIXES:
        return name
    raise HTTPException(
        status_code=400,
        detail=f"Unknown voice '{name}'. OpenAI voices: {list(VOICE_MAP.keys())}. "
        f"Or use a Kokoro voice name directly (e.g. af_heart, am_adam).",
    )


def lang_code_for_voice(voice: str) -> str:
    """Derive the language code from the voice name prefix."""
    prefix = voice[0]
    return LANG_PREFIXES.get(prefix, "a")


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    voice = resolve_voice(req.voice)
    lang_code = lang_code_for_voice(voice)
    pipeline = get_pipeline(lang_code)

    logger.info(
        f"TTS: voice={voice} lang={lang_code} speed={req.speed} "
        f"fmt={req.response_format} chars={len(req.input)}"
    )

    t0 = time.perf_counter()
    chunks = []
    for _gs, _ps, audio in pipeline(req.input, voice=voice, speed=req.speed):
        chunks.append(audio)

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio generated.")

    samples = np.concatenate(chunks)
    elapsed = time.perf_counter() - t0
    duration = len(samples) / SAMPLE_RATE
    logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")

    audio_bytes = encode_audio(samples, req.response_format)
    return Response(
        content=audio_bytes,
        media_type=CONTENT_TYPES[req.response_format],
        headers={"Content-Disposition": f'inline; filename="speech.{req.response_format}"'},
    )


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded.")

    # Write to temp file — faster-whisper requires a file path
    suffix = os.path.splitext(file.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()

        t0 = time.perf_counter()
        kwargs: dict = {}
        if language:
            kwargs["language"] = language
        segments, info = whisper_model.transcribe(tmp.name, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.perf_counter() - t0

        logger.info(
            f"STT: lang={info.language} prob={info.language_probability:.2f} "
            f"duration={info.duration:.1f}s elapsed={elapsed:.2f}s"
        )

        if response_format == "verbose_json":
            return {
                "text": text,
                "language": info.language,
                "duration": info.duration,
            }
        return {"text": text}
    finally:
        os.unlink(tmp.name)


@app.get("/v1/audio/voices")
async def list_voices():
    """List all available voices (not part of OpenAI API, but useful)."""
    return {
        "openai_mapped": VOICE_MAP,
        "kokoro_voices": [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
            "am_michael", "am_onyx", "am_puck", "am_santa",
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
            "ef_dora", "em_alex", "em_santa",
            "ff_siwis",
            "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
            "if_sara", "im_nicola",
            "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
            "pf_dora", "pm_alex", "pm_santa",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
        ],
    }


@app.get("/v1/models")
async def list_models():
    """Minimal /v1/models so clients can discover this server."""
    return {
        "object": "list",
        "data": [
            {
                "id": "kokoro",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            },
            {
                "id": "tts-1",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            },
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tts_backend": "kokoro-pytorch-gpu",
        "stt_backend": "faster-whisper-gpu",
        "stt_model": "large-v3",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8880)
