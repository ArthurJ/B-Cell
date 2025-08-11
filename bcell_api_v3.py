import json
import os
from itertools import islice, cycle
from typing import List, Annotated, Optional

import logfire
import tempfile
from aiofiles.tempfile import NamedTemporaryFile
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from secrets import token_hex

from pydantic import BaseModel
from pydantic_ai.agent import AgentRunResult

from dialog import interaction, tts, transcribe, initial_run, DialogContext, chorus

app = FastAPI(title='B-Cell API V3')
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["https://audiofusion.com.br", "https://audiofb.com"], Definir posteriormente a URL para acesso.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logfire.configure(service_name="API_v3", scrubbing=False)
logfire.instrument_fastapi(app)
logfire.instrument_system_metrics()


class Chat(BaseModel):
    thread_id: str
    history: List
    deps: DialogContext
    last_text: str
    sources: Optional[List[str]]

claims = json.load(open("knowledge/talvey-claims.json", 'r'))
chats = dict()


async def save_audios(source_audio, qtd_voices=3):
    voices = await chorus(source_audio, qtd_voices=qtd_voices)
    audio_list = []

    for d in islice(cycle(voices), 7):
        async with NamedTemporaryFile(suffix='.mp3',
                                      delete_on_close=False,
                                      delete=False) as audio_file:
            await audio_file.write(d)
            logfire.info(f'Audio saved: {audio_file.name}')
            audio_list.append(audio_file.name.split('/')[-1])
    return audio_list


def update_chat(chat: Chat, result: AgentRunResult):
    chat.history = result.all_messages()
    chat.last_text = result.output.answer
    chat.sources = [''.join(s[-1::-1].split('.')[1:])[-1::-1].split('/')[-1] for s in result.output.sources]

@app.get("/new-chat")
@app.get("/v3/new-chat")
async def new_chat(lang:str='en'):
    chat_id = token_hex()
    deps = DialogContext(talvey_claims=claims)
    first_run = await initial_run(deps)
    chat = Chat(thread_id=chat_id,
                history=first_run.all_messages(),
                deps=deps,
                last_text=first_run.output.answer,
                sources=None)
    chats[chat_id] = chat
    logfire.info(f'Chat created: {chat_id} handled by worker PID: {os.getpid()}')
    return {"chat_id": chat_id, 'ai_message':first_run.output.answer, 'sources':[]}


@app.get("/chat/text/{chat_id}")
@app.get("/v3/chat/text/{chat_id}")
async def send_text(chat_id:str, message:str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    message = BeautifulSoup(message, "html.parser").get_text()
    result = (await interaction(message, chat.deps, chat.history))
    update_chat(chat, result)

    return {'ai_message': result.output.answer, 'sources': result.output.sources}


@app.get("/chat/mixed/{chat_id}")
@app.get("/chat/v2/mixed/{chat_id}")
@app.get("/v3/chat/mixed/{chat_id}")
async def send_mixed(chat_id: str, message: str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]

    message = BeautifulSoup(message, "html.parser").get_text()
    result = (await interaction(message, chat.deps, chat.history))
    update_chat(chat, result)

    source_audio = await tts(result.output.answer)

    audio_file_list = await save_audios(source_audio)
    return JSONResponse(audio_file_list)

@app.post("/chat/audio/{chat_id}")
@app.post("/chat/v2/audio/{chat_id}")
@app.post("/v3/chat/audio/{chat_id}")
async def send_audio(chat_id:str,
                     audio: Annotated[UploadFile,
                     File(description="filetypes: flac, m4a, mp3, wav, webm")]=None):
    if not audio:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]

    suffix = '.' + audio.filename.split('.')[-1]
    try:
        contents = await audio.read()
        async with (NamedTemporaryFile(suffix=suffix) as f):
            await f.write(contents)
            logfire.info(f'Audio received: {f.name}')
    except Exception as e:
        logfire.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f'Something went wrong:\n{str(e)}')
    finally:
        audio.file.close()

    transcription = await transcribe(contents, audio_type=f'audio/{suffix}')
    result = (await interaction(transcription, chat.deps, chat.history))
    update_chat(chat, result)

    source_audio = await tts(result.output.answer)

    audio_file_list = await save_audios(source_audio)
    return JSONResponse(audio_file_list)

@app.get("/chat/v2/download/{file_name}")
@app.get("/v3/chat/download/{file_name}")
async def download_audio(file_name:str):
    f_name = os.path.join(tempfile.gettempdir(), file_name)
    if os.path.basename(file_name) != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if not os.path.exists(f_name):
        return HTTPException(status_code=404, detail=f'File not found.')
    if not os.path.isfile(f_name):
        return HTTPException(status_code=404, detail=f'File not found.')
    if not os.path.isabs(f_name):
        return HTTPException(status_code=404, detail=f'File not found.')
    with open(f_name, 'rb') as f:
        return FileResponse(f.name, media_type='audio/mpeg')


@app.get("/chat/last-text/{chat_id}")
@app.get("/chat/v2/last-text/{chat_id}")
@app.get("/v3/chat/last-text/{chat_id}")
async def get_last_message(chat_id:str):
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    logfire.info(f'Sources used: {chat.sources}')
    return {'ai_message': chat.last_text, 'sources': chat.sources}