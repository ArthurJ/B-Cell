from collections import deque
from tempfile import NamedTemporaryFile
from typing import Annotated
import logging

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from secrets import token_hex

from langchain_core.chat_history import InMemoryChatMessageHistory
from dialogue import text_interaction, audio_interaction

app = FastAPI(title='API')
LOG = logging.getLogger('uvicorn.error')
chats = dict()

class Chat(BaseModel):
    thread_id: str
    context: deque
    language: str
    memory: InMemoryChatMessageHistory

@app.get("/")
def read_root():
    raise HTTPException(status_code=404)

@app.get("/new-chat")
async def new_chat(lang:str = 'English'):
    chat_id = token_hex()
    chat = Chat(thread_id=chat_id,
                context=deque(maxlen=20),
                language=lang,
                memory=InMemoryChatMessageHistory())
    chats[chat_id] = chat
    LOG.info(chat_id)
    return {"chat_id": chat_id}

@app.get("/chat/text/{chat_id}")
async def send_text(chat_id:str, message:str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    output = text_interaction(message,
                              {"configurable": {"thread_id": chat_id}},
                              chat.context,
                              chat.language,
                              chat.memory)
    return {'ai_message': output}

@app.post("/chat/audio/{chat_id}")
async def send_audio(chat_id:str,
                     audio: Annotated[UploadFile,
                     File(description="filetypes: flac, m4a, mp3, mp4, mpeg, oga, ogg, wav, webm")]=None):
    if not audio:
        return

    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]

    suffix = '.' + audio.filename.split('.')[-1]
    try:
        contents = audio.file.read()
        with NamedTemporaryFile(suffix=suffix) as f:
            f.write(contents)
            LOG.info(f.name)
            output = audio_interaction(f.name,
                                       {"configurable": {"thread_id": chat_id}},
                                       chat.context,
                                       chat.language,
                                       chat.memory)
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

