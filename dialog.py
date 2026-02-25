import asyncio
import json

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import logfire
from dotenv import load_dotenv
from pydantic import Field, BaseModel, conlist
from pydantic_ai import Agent, RunContext, BinaryContent

from codetiming import Timer

# import simpleaudio as sa

import base64
from openai import OpenAI

from agent_tools import tools
from voice import gather_voices, pcm_2_wav

logfire.configure(service_name="dialog", scrubbing=False)
logfire.instrument_openai()

load_dotenv()
openai_client = OpenAI()

# main_model='openai:gpt-4o'
main_model='openai:gpt-5.2-2025-12-11'
secundary_model='openai:gpt-4o-mini'
audio_model='openai:gpt-4o-mini-audio-preview-2024-12-17'

#Language = Literal['en', 'pl', 'pt']
Language = Literal['en']

@dataclass
class DialogContext:
    talvey_claims: str
    system_prompt: str
    target_language: Language

class MainAgentOutputType(BaseModel):
    footnotes: conlist(str, min_length=0)
    answer: str = Field(description="The answer to the user's message. Do not rely on your prior knowledge to write your answers; always justify it with the `knowledge_retrieve` tool.")
    sources: conlist(str) = Field(description='List of **sources** used to write the answer.'
                                              '**Always** support your answers with relevant sources in the sources field.'
                                              "If a question seems too simple, expand the answer with depth and context that require citations."
                                              "That's critical, the answer **must** have at least one valid (and not null) source. "
                                  ,validation_alias='metadata.source')
    is_source_relevant: bool = Field(description='Whether or not the given sources are relevant to the user learn more about the question they made.')

class TranscriberOutputType(BaseModel):
    original_transcription: str = Field(
        description="The raw transcription. Apply phonetic corrections (e.g., 'beatles' -> 'B-Cells') here.")
    language: str = Field(description="ISO 639-1 language code.")
    english_transcription: str = Field(
        description="The final English translation/transcription. Ensure medical terminology is accurate.")

class TranslatorOutputType(BaseModel):
    translation: str = Field(
        description=f"The translation to the target language. Ensure medical terminology is accurate.")

transcriber = Agent(
    retries=3,
    instrument=True,
    output_type=TranscriberOutputType,
    instructions=Path('transcriber_prompt.md').read_text()
)

translator = Agent(
    model=main_model,
    retries=3,
    instrument=True,
    output_type=TranslatorOutputType,
    deps_type=DialogContext,
    instructions=Path('translator_prompt.md').read_text()
)

bcell = Agent(
    model=main_model,
    deps_type=DialogContext,
    tools=tools,
    output_type=MainAgentOutputType,
    retries=3,
    instrument=True
)

@bcell.system_prompt
def add_claims(ctx: RunContext[DialogContext]) -> str:
    return f'TALVEY SmPC (EU), 2025:\n{ctx.deps.talvey_claims}'

@bcell.system_prompt
def add_prompt(ctx: RunContext[DialogContext]) -> str:
    return f'Your prompt:\n{ctx.deps.system_prompt}'

@translator.system_prompt
def language(ctx: RunContext[DialogContext]) -> str:
    return f'Target language:\n{ctx.deps.target_language}'

async def interaction(query: str, dependencies: DialogContext,
                      chat_history, model=main_model):
    to_english_context = DialogContext(target_language='en', talvey_claims='', system_prompt='')
    with Timer(initial_text='\nMain model', logger=logfire.info):
        query = (await translate(query, to_english_context))
        b_cell_result = await bcell.run(query, message_history=chat_history, deps=dependencies, model=model)

    return b_cell_result

async def translate(query: str, dependencies: DialogContext) -> str:
    return (await translator.run(user_prompt=query, deps=dependencies)).output.translation

async def transcribe(audio: bytes, audio_type='audio/mp3') -> str:
    return (await transcriber.run([BinaryContent(audio, media_type=audio_type)], model=audio_model)).output.english_transcription

async def tts(text: str) -> bytes:
    cleaned_text = (
        text
        .replace("TALVEY®▼ (talquetamab)", "TALVEY")
        .replace("CARVYKTI®▼ (ciltacabtagene autoleucel)", "CARVYKTI")
        .replace("ABECMA®▼ (idecabtagene vicleucel)", "ABECMA")
        .replace("ELREXFIO®▼ (elranatamab)", "ELREXFIO")
        .replace("TECVAYLI®▼ (teclistamab)", "TECVAYLI")
        .replace("DARZALEX® (daratumumab)", "DARZALEX")
        .replace("®", "")
        .replace("▼", "")
        .replace("TECVAYLI", "TECVAYLEE")
        .replace("GPRC5D", "G-P-R-see-five-D")
        .replace("SmPC", "S-M-P-C")
    )

    with Timer(initial_text='\nText-To-Speech', logger=logfire.info):
        completion = openai_client.chat.completions.create(
            model="gpt-audio-mini",
            modalities=["text", "audio"],
            audio={"voice": "fable", "format": "pcm16"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        *Don't act as conversational assistant.*
                        You are a **british** voice actor.
                        Your character's voice sounds ethereal and wise, but also helpful and collaborative.
                                                
                        CRITICAL RULES:
                        1. Do NOT use conversational fillers (e.g., 'Sure', 'Of course', 'Here is the text', 'Understood').
                        2. Start reading the provided text IMMEDIATELY.
                        3. Do not improvise or add words that are not in the text.
                        4. PACE: Speak calmly, and deliberately. Pause for effect at punctuation.
                        
                        PRONUNCIATION GUIDES:
                        - "myeloma": [my-uh-LOH-muh]
                        - "MonumenTAL-1": [Mohn-you-ment-tahl-One]
                        - "CARVYKTI": [car-vick-tee]
                        - "TALQUETAMAB": [tal-kwet-a-mab]
                        - "regimen": Do NOT pronounce as "regiment".
                        
                        Your task is to read the text provided by the user aloud.
                        """
                    )
                },
                {
                    "role": "user",
                    "content": f"Read the following text exactly as written:\n\n{cleaned_text}"
                }
            ]
        )
    return base64.b64decode(completion.choices[0].message.audio.data)

async def chorus(pcm_audio:bytes, qtd_voices=1, play=False, convert=True) -> List[bytes]:
    wav_data = pcm_2_wav(pcm_audio)
    if play:
        pass
       # data = await gather_voices(wav_data, 'pcm_44100', 1)
       # sa.play_buffer(data[0],1, sample_rate=44100, bytes_per_sample=2)
    if convert:
        with Timer(initial_text='\nGathering Voices', logger=logfire.info):
            data = await gather_voices(wav_data, qtd_voices=qtd_voices)
            return data
    return []

async def initial_run(deps: DialogContext):
    return await bcell.run("Introduce yourself.", deps=deps,
                           model='openai:gpt-4o-mini')

def tts_sync(text:str) -> bytes:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(tts(text))

def initial_run_sync(deps: DialogContext):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(initial_run(deps))

def interaction_sync(query, dependencies: DialogContext, chat_history, model):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(interaction(query, dependencies, chat_history, model))

if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(transcribe(open('/home/arthur/Documents/query.mp3', 'rb').read()))

    deps = DialogContext(
        talvey_claims= json.load(open("knowledge/talvey-claims.json", 'r')),
        system_prompt= open('system_prompt.md', 'r').read(),
        target_language='pt'
    )

    initial = initial_run_sync(deps)
    history = initial.all_messages()
    print(initial.output.answer)
    loop = asyncio.get_event_loop()

    # ## Text loop
    while True:
        query_input = input('User: ')
        print()
        result = interaction_sync(query_input, deps, history, model=main_model)
        history = result.all_messages()
        print(result.all_messages()[-1])
        print(result.output.answer)
        # loop.run_until_complete(chorus(tts_sync(result.output.answer), play=False, convert=True))
        sources = result.output.sources
        print(sources)

    ## Audio loop
    ## pass

    ## Batch text
    # configs = [
    #             ('openai:gpt-4o', None, 1),
    #             # ('openai:gpt-4o', 'openai:gpt-4o', 1),
    #             # ('openai:gpt-4o', 'openai:gpt-4o', 3),
    #             # ('openai:gpt-4.1', None, 1),
    #             # ('openai:gpt-4.1', 'openai:gpt-4.1', 1),
    #             # ('openai:gpt-4.1', 'openai:gpt-4.1', 3),
    #             # ('openai:gpt-5', None, 1),
    #             # ('openai:gpt-5', 'openai:gpt-5', 1),
    #             # ('openai:gpt-5', 'openai:gpt-5', 3),
    #            ]
    #
    # queries = [
    #     'Quais pacientes não são elegíveis ao tratamento com talvey?',
    #     'Quais pacientes são elegíveis ao tratamento com talvey?',
    #     'Qual a justificativa biológica para que o tratamento com talvey preservar as células B?',
    #     'Como faço torta de manga? Me passa a receita por gentileza?'
    # ]
    #
    # for i in product(configs, queries):
    #     print('\n', i, '\n')
    #     (model, sec_model, passes), query = i
    #
    #     result = interaction_sync(query, deps, history, model, sec_model, passes)
    #     print(result.output.answer)
    #     loop.run_until_complete(chorus(tts_sync(result.output.answer), play=False, convert=True))
    #     print(result.output.sources)
