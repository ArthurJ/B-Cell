import asyncio
import io
import os
import wave
from random import sample
from typing import Tuple

from elevenlabs.client import AsyncElevenLabs

elevenlabs_client = AsyncElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

async def convert_voice(audio_bytes, voice_id, out_format='pcm_44100'):
    converted = elevenlabs_client.speech_to_speech.convert(
        audio=audio_bytes,
        voice_id=voice_id,
        model_id="eleven_multilingual_sts_v2",
        output_format=out_format,
    )

    converted_audio_bytes = b""
    async for chunk in converted:
        if chunk:
            converted_audio_bytes += chunk

    return converted_audio_bytes

async def gather_voices(audio_bytes, out_format='mp3_44100_192', qtd_voices=3):
    voice_ids = ['ELEVEN_LABS_VOICE_ID_1', 'ELEVEN_LABS_VOICE_ID_2',
                 'ELEVEN_LABS_VOICE_ID_3', 'ELEVEN_LABS_VOICE_ID_4',
                 'ELEVEN_LABS_VOICE_ID_5', 'ELEVEN_LABS_VOICE_ID_6',
                 'ELEVEN_LABS_VOICE_ID_7']
    ids = sample(voice_ids, k=qtd_voices)

    return await asyncio.gather(
        *(convert_voice(audio_bytes, os.getenv(i), out_format) for i in ids)
    )


def pcm_2_wav(pcm_data: bytes, channels: int = 1,
              sample_width: int = 2, frame_rate: int = 24000) -> bytes:
    in_memory_file = io.BytesIO()
    with wave.open(in_memory_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        wf.writeframes(pcm_data)

    in_memory_file.seek(0)
    return in_memory_file.read()


def wav_2_pcm(wav_data: bytes) -> Tuple[bytes, int, int, int]:
    in_memory_file = io.BytesIO(wav_data)
    with wave.open(in_memory_file, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

        return (pcm_data, channels, sample_width, frame_rate)
