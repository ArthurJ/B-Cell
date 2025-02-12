from collections import deque
from tempfile import NamedTemporaryFile
from typing import Annotated
import logging

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from secrets import token_hex

from dialogue import text_interaction, audio_interaction, audio_stream

app = FastAPI(title='API')
LOG = logging.getLogger('uvicorn.error')
chats = dict()

class Chat(BaseModel):
    thread_id: str
    context: deque
    language: str

@app.get("/")
def read_root():
    raise HTTPException(status_code=404)

@app.get("/new-chat")
async def new_chat(lang:str = 'English'):
    thread_id = token_hex()
    chat = Chat(thread_id=thread_id,
                context=deque(maxlen=3),
                language=lang)
    chats[thread_id] = chat
    LOG.info(thread_id)
    return {"thread_id": thread_id}

@app.post("/chat/text/{thread}")
async def send_text(thread:str, message:str):
    if not message:
        return
    if thread not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[thread]
    output = text_interaction(message,
                              {"configurable": {"thread_id": thread}},
                              chat.context,
                              chat.language)
    return {'ai_message': output}

@app.post("/chat/audio/{thread}")
async def send_audio(thread:str,
                     audio: Annotated[UploadFile, File(description="filetypes: flac, m4a, mp3, mp4, mpeg, oga, ogg, wav, webm")]=None):
    if not audio:
        return

    if thread not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[thread]

    suffix = '.' + audio.filename.split('.')[-1]
    try:
        contents = audio.file.read()
        with NamedTemporaryFile(suffix=suffix) as f:
            f.write(contents)
            LOG.info(f.name)
            output = audio_interaction(f.name,
                                       {"configurable": {"thread_id": thread}},
                                       chat.context,
                                       chat.language)
    except Exception as e:
        LOG.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        audio.file.close()

    with NamedTemporaryFile(suffix=suffix,
                            delete_on_close=False,
                            delete=False) as audio_file:
        audio_file.write(b''.join(output))
        LOG.info(f.name)
        return FileResponse(audio_file.name)

@app.post("/stream-chat/text/{thread}")
async def stream_text(thread:str, message:str):
    if not message:
        return
    if thread not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[thread]
    return StreamingResponse(text_interaction(message,
                             {"configurable": {"thread_id": thread}},
                             chat.context,
                             chat.language)
    )

@app.post("/stream-chat/audio/{thread}")
async def stream_audio(thread:str,
                     audio: Annotated[UploadFile, File(description="filetypes: flac, m4a, mp3, mp4, mpeg, oga, ogg, wav, webm")]=None):
    if not audio:
        return

    if thread not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[thread]

    suffix = '.' + audio.filename.split('.')[-1]
    try:
        contents = audio.file.read()
        with NamedTemporaryFile(suffix=suffix) as f:
            f.write(contents)
            LOG.info(f.name)
            return StreamingResponse(audio_stream(f.name,
                                       {"configurable": {"thread_id": thread}},
                                       chat.context,
                                       chat.language),  media_type=f"audio/mp3")
    except Exception as e:
        LOG.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        audio.file.close()