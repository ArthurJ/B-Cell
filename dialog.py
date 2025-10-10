import asyncio
import json

from dataclasses import dataclass
from typing import List

import logfire
from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent, RunContext, BinaryContent

# import simpleaudio as sa

import base64
from openai import OpenAI

from agent_tools import tools
from voice import gather_voices, pcm_2_wav

logfire.configure(service_name="dialog", scrubbing=False)
logfire.instrument_openai()

load_dotenv()
openai_client = OpenAI()

@dataclass
class DialogContext:
    talvey_claims: str
    system_prompt: str

@dataclass
class JudgeOutputType:
    critique: str
    adjusted_answer: str

@dataclass
class MainAgentOutputType:
    answer: str
    sources: List[str] = Field(validation_alias='metadata.source')

@dataclass
class TranscriberOutputType:
    original_transcription: str
    language: str
    english_transcription: str

transcriber = Agent(
    model='google-gla:gemini-2.5-flash',
    retries=3,
    instrument=True,
    output_type=TranscriberOutputType,
    instructions='You are an excellent Captioner, Transcriptionist and Translator.'
                 "Some of the queries you'll receive are questions, treat it as any other sentence:"
                 'Transcribe, translating to english if necessary.'
                 'Your only jobs is to transcribe and translate, nothing else.'
)

judge = Agent(
    model='openai:gpt-4o',
    deps_type=DialogContext,
    tools=tools,
    output_type=JudgeOutputType,
    retries=3,
    instrument=True,
    system_prompt='You are the Judge agent, your responsibility is to evaluate the answer of another agent (B-Cell), '
                  'and adjust it\'s answer to ensure that it is sticking with the rules and standards provided.',
    instructions="""
                 Considering the **persona**, the **correctness** and **completeness** of the response, 
                 and **compliance** with the safety rules.
                 You have the same tools and information available to B-Cell agent. 
                 Please, critique the response and offer a corrected version of it.
                 """
)

bcell = Agent(
    model='openai:gpt-4o',
    deps_type=DialogContext,
    tools=tools,
    output_type=MainAgentOutputType,
    retries=3,
    instrument=True
)

@bcell.system_prompt
def add_claims(ctx: RunContext[DialogContext]) -> str:
    return f'Talvey Claims:\n{ctx.deps.talvey_claims}'

@bcell.system_prompt
def add_prompt(ctx: RunContext[DialogContext]) -> str:
    return f'Your prompt:\n{ctx.deps.system_prompt}'

@judge.system_prompt
def add_claims(ctx: RunContext[DialogContext]) -> str:
    return f'Talvey Claims:\n{ctx.deps.talvey_claims}'

@judge.system_prompt
def add_prompt(ctx: RunContext[DialogContext]) -> str:
    return f'B-Cells prompt:\n{ctx.deps.system_prompt}'


async def interaction(query: str, dependencies: DialogContext, chat_history):
    b_cell_result = await bcell.run(
        (await transcriber.run(query)).output.english_transcription,
        message_history=chat_history,
        deps=dependencies,
    )
    adjustment = (await judge.run(
        [f'User query: {query}',
                    f'B-Cell answer: {b_cell_result.output.answer}',
                    f'Provided sources:{b_cell_result.output.sources}'],
        deps=dependencies,
    )).output.adjusted_answer
    b_cell_result.output.answer = adjustment

    return b_cell_result

async def transcribe(audio: bytes, audio_type='audio/mp3') -> str:
    return (await transcriber.run([BinaryContent(audio, media_type=audio_type)])).output.english_transcription


async def tts(text:str) -> bytes:
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=[
            {
                "role": "system",
                "content": "You are a voice actor."
                           "Do not improvise."
                           "Your character's voice sounds ethereal and wise, but also helpful and collaborative."
            },
            {
                "role": "user",
                "content": f'Say: {text}'
            }
        ]
    )
    return  base64.b64decode(completion.choices[0].message.audio.data)

async def chorus(pcm_audio:bytes, qtd_voices=1, play=False, convert=True) -> List[bytes]:
    wav_data = pcm_2_wav(pcm_audio)
    # if play:
    #    data = await gather_voices(wav_data, 'pcm_44100', 1)
    #    sa.play_buffer(data[0],1, sample_rate=44100, bytes_per_sample=2)
    if convert:
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

def interaction_sync(query, dependencies: DialogContext, chat_history):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(interaction(query, dependencies, chat_history))

if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(transcribe(open('/home/arthur/Documents/query.mp3', 'rb').read()))

    deps = DialogContext(
        talvey_claims= json.load(open("knowledge/talvey-claims.json", 'r')),
        system_prompt= open('system_prompt.md', 'r').read()
    )

    initial = initial_run_sync(deps)
    history = initial.all_messages()
    print(initial.output.answer)
    loop = asyncio.get_event_loop()

    while True:
        query_input = input('User: ')
        print()
        result = interaction_sync(query_input, deps, history)
        history = result.all_messages()
        print(result.all_messages()[-1])
        print(result.output.answer)
        # loop.run_until_complete(chorus(tts_sync(result.output.answer), play=False, convert=False))
        sources = result.output.sources
        sources = [''.join(s[-1::-1].split('.')[1:])[-1::-1].split('/')[-1] for s in sources]
        print(sources)
