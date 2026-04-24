import json
import os
from itertools import islice, cycle
from pathlib import PurePath
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

from dialog import interaction, tts, transcribe, initial_run, DialogContext, chorus, Language, translate

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
    sources: List[str]
    footnotes: List[str]

class TextResponse(BaseModel):
    ai_message: str
    footnotes: List[str]
    sources: List[str]


claims = json.load(open("knowledge/talvey-claims.json", 'r'))
prompt= open('system_prompt.md', 'r').read()

chats = dict()

# Mapeamento de fontes especificas por idioma/regiao
REGIONAL_SOURCE_OVERRIDES = {
    'pl': {
        'Rasche L et al. Presented at: American Society of Clinical Oncology (ASCO) Congress; 30 May--3 June 2025.': 'Rasche, L. et al. Presented at: ASCO 2025 Annual Meeting; 30 May–-Jun 03 2025 (Abstract no. 7528). Available at: https://ascopubs.org/doi/10.1200/JCO.2025.43.16_suppl.7528 (last accessed April 2026).',
        'Rasche L et al. Presented at: American Society of Clinical Oncology (ASCO) Congress; 30 May-3 June 2025.': 'Rasche, L. et al. Presented at: ASCO 2025 Annual Meeting; 30 May–-Jun 03 2025 (Abstract no. 7528). Available at: https://ascopubs.org/doi/10.1200/JCO.2025.43.16_suppl.7528 (last accessed April 2026).',
        'Rasche L et al. Presented at: American Society of Clinical Oncology (ASCO) Congress; 30 May–3 June 2025; Chicago, IL (Poster no. 96).': 'Rasche, L. et al. Presented at: ASCO 2025 Annual Meeting; 30 May–-Jun 03 2025 (Abstract no. 7528). Available at: https://ascopubs.org/doi/10.1200/JCO.2025.43.16_suppl.7528 (last accessed April 2026).',

        'Rasche L et al. Presented at: European Hematology Association (EHA) Congress; 13--16 June 2024; Madrid, Spain (Poster no. P915).': 'Rasche, L. et al. Presented at: EHA 2024 Hybrid Congress; 13–16 June (Abstract no. P915). Available at: https://library.ehaweb.org/eha/2024/eha2024-congress/420979/leo.rasche.long-term.efficacy.and.safety.results.from.the.phase.1.2.html (last accessed April 2026).',
        'Rasche L et al. Presented at: European Hematology Association (EHA) Congress; 13–16 June 2024; Madrid, Spain.': 'Rasche, L. et al. Presented at: EHA 2024 Hybrid Congress; 13–16 June (Abstract no. P915). Available at: https://library.ehaweb.org/eha/2024/eha2024-congress/420979/leo.rasche.long-term.efficacy.and.safety.results.from.the.phase.1.2.html (last accessed April 2026).',
        'Rasche L et al. Presented at: European Hematology Association (EHA) Congress; 13–16 June 2024; Madrid, Spain (Poster no. P915).': 'Rasche, L. et al. Presented at: EHA 2024 Hybrid Congress; 13–16 June (Abstract no. P915). Available at: https://library.ehaweb.org/eha/2024/eha2024-congress/420979/leo.rasche.long-term.efficacy.and.safety.results.from.the.phase.1.2.html (last accessed April 2026).'
    }
}

def apply_source_overrides(source: str, language: str) -> str:
    overrides = REGIONAL_SOURCE_OVERRIDES.get(language, {})
    
    for original, replacement in overrides.items():
        if original in source:
            return source.replace(original, replacement)
            
    return source


async def save_audios(source_audio, qtd_voices=3, language:Language='en'):
    voices = await chorus(source_audio, qtd_voices=qtd_voices, language=language)
    audio_list = []

    for d in islice(cycle(voices), 7):
        async with NamedTemporaryFile(suffix='.mp3',
                                      delete_on_close=False,
                                      delete=False) as audio_file:
            await audio_file.write(d)
            logfire.info(f'Audio saved: {audio_file.name}')
            audio_list.append(audio_file.name.split('/')[-1])
    return audio_list

def format_reference(p):
    path = PurePath(p)
    if path.suffix.lower() in ('.pdf', '.md'):
        return path.stem
    return p


async def update_chat(chat: Chat, result: TextResponse):
    ai_message = ''
    footnotes = []
    if chat.deps.target_language != 'en':
        ai_message = await translate(result.output.answer, chat.deps)
        footnotes = [await translate(note, chat.deps) for note in result.output.footnotes]

    chat.history = result.all_messages()
    chat.last_text = ai_message or result.output.answer
    chat.footnotes = footnotes or result.output.footnotes
    
    raw_sources = {f"{format_reference(p).rstrip('.')}."
                   for p in (result.output.sources or [])
                   if result.output.is_source_relevant}
    
    overridden_sources = [apply_source_overrides(src, chat.deps.target_language) for src in raw_sources]
    chat.sources = list(set(overridden_sources))

@app.get("/new-chat")
async def new_chat(lang:Language='en'):
    chat_id = token_hex()
    deps = DialogContext(talvey_claims=claims, system_prompt=prompt, target_language=lang)
    chat = Chat(thread_id=chat_id,
                history=[],
                deps=deps,
                last_text="",
                sources=[],
                footnotes=[])
    chats[chat_id] = chat
    logfire.info(f'Chat created: {chat_id} handled by worker PID: {os.getpid()}')
    return {"chat_id": chat_id, 'ai_message':"", 'sources':[]}


@app.get("/chat/text/{chat_id}")
@app.get("/chat/v2/text/{chat_id}")
async def send_text(chat_id:str, message:str) -> TextResponse:
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    message = BeautifulSoup(message, "html.parser").get_text()
    result = (await interaction(message, chat.deps, chat.history))
    await update_chat(chat, result)

    return TextResponse(ai_message=chat.last_text, sources=chat.sources, footnotes=chat.footnotes)

@app.get("/chat/mixed/{chat_id}")
@app.get("/chat/v2/mixed/{chat_id}")
async def send_mixed(chat_id: str, message: str):
    if not message:
        return
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]

    message = BeautifulSoup(message, "html.parser").get_text()
    result = (await interaction(message, chat.deps, chat.history))
    await update_chat(chat, result)

    source_audio = await tts(chat.last_text, language=chat.deps.target_language)

    audio_file_list = await save_audios(source_audio, language=chat.deps.target_language)
    return JSONResponse(audio_file_list)

@app.post("/chat/audio/{chat_id}")
@app.post("/chat/v2/audio/{chat_id}")
async def send_audio(chat_id:str,
                     audio: Annotated[UploadFile,
                     File(description="filetypes: mp3, wav")]=None):
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

    transcription = await transcribe(contents, audio_type=audio.content_type)
    result = (await interaction(transcription, chat.deps, chat.history))
    await update_chat(chat, result)

    source_audio = await tts(chat.last_text, language=chat.deps.target_language)

    audio_file_list = await save_audios(source_audio, language=chat.deps.target_language)
    return JSONResponse(audio_file_list)

@app.get("/chat/v2/download/{file_name}")
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
async def get_last_message(chat_id:str)-> TextResponse:
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    chat: Chat = chats[chat_id]
    # O texto ja foi traduzido no update_chat, entao so precisamos retorna-lo
    return TextResponse(ai_message=chat.last_text, sources=chat.sources, footnotes=chat.footnotes)

@app.post("/test/dictate")
async def dictate(phrase:str):
    source_audio = await tts(phrase)
    audio_file_list = await save_audios(source_audio)
    return JSONResponse(audio_file_list)