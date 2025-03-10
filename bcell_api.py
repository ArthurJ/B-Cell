from collections import deque
from tempfile import NamedTemporaryFile
from typing import Annotated
import logging

from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse

from pydantic import BaseModel

from secrets import token_hex

from langchain_core.chat_history import InMemoryChatMessageHistory
from dialogue import text_interaction, audio_interaction, mixed_interaction

app = FastAPI(title='B-Cell API')

uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.propagate = False
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(levelname)-5s\t%(message)s",
    handlers=[logging.StreamHandler()],
)
app_logger = logging.getLogger(__name__)

chats = dict()

class Chat(BaseModel):
    thread_id: str
    context: deque
    language: str
    memory: InMemoryChatMessageHistory

@app.get("/new-chat")
def new_chat(lang:str = 'en'):
    chat_id = token_hex()
    chat = Chat(thread_id=chat_id,
                context=deque(maxlen=3),
                language=lang,
                memory=InMemoryChatMessageHistory())
    chats[chat_id] = chat
    app_logger.info(f'Chat created: {chat_id}')
    return {"chat_id": chat_id}

@app.get("/chat/text/{chat_id}")
def send_text(chat_id:str, message:str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    message = BeautifulSoup(message, "html.parser").get_text()
    output = text_interaction(message,
                              {"configurable": {"thread_id": chat_id}},
                              chat.context,
                              chat.language,
                              chat.memory)
    return {'ai_message': output}

@app.get("/chat/last-text/{chat_id}")
def get_last_message(chat_id:str):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    return {'ai_message': chat.memory.messages[-1].content}

@app.get("/chat/mixed/{chat_id}")
def send_text(chat_id:str, message:str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]

    message = BeautifulSoup(message, "html.parser").get_text()
    audio_output = mixed_interaction(message,
                                     {"configurable": {"thread_id": chat_id}},
                                     chat.context,
                                     chat.language,
                                     chat.memory)
    with NamedTemporaryFile(suffix='.mp3',
                            delete_on_close=False,
                            delete=False) as audio_file:
        audio_file.write(b''.join(audio_output))
        app_logger.info(f'Audio sent: {audio_file.name}')
        return  FileResponse(audio_file.name, media_type='audio/mpeg')

@app.post("/chat/audio/{chat_id}")
def send_audio(chat_id:str,
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
        with (NamedTemporaryFile(suffix=suffix) as f):
            f.write(contents)
            app_logger.info(f'Audio received: {f.name}')
            audio_output = audio_interaction(f.name,
                                             {"configurable": {"thread_id": chat_id}},
                                             chat.context,
                                             chat.language,
                                             chat.memory)
    except Exception as e:
        app_logger.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f'Something went wrong:\n{str(e)}')
    finally:
        audio.file.close()

    with NamedTemporaryFile(suffix='.mp3',
                            delete_on_close=False,
                            delete=False) as audio_file:
        audio_file.write(b''.join(audio_output))
        app_logger.info(f'Audio sent: {audio_file.name}')
        return FileResponse(audio_file.name, media_type='audio/mpeg')

