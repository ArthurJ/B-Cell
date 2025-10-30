import asyncio
import json
# from itertools import product


from dataclasses import dataclass
from typing import List, Optional

import logfire
from dotenv import load_dotenv
from pydantic import Field
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

judge_model='openai:gpt-4o'
main_model='openai:gpt-4o'

@dataclass
class DialogContext:
    talvey_claims: str
    system_prompt: str

@dataclass
class JudgementCriteriaType:
    compliance: bool
    compliance_short_critique: Optional[str]
    persona_adherence: bool
    persona_adherence_short_critique: Optional[str]
    correctness: bool
    correctness_short_critique: Optional[str]
    completeness: bool

@dataclass
class JudgementType:
    criteria: JudgementCriteriaType
    adjusted_answer: str
    sources: List[str] = Field(validation_alias='metadata.source')

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
    model=judge_model,
    deps_type=DialogContext,
    tools=tools,
    output_type=JudgementType,
    retries=3,
    instrument=True,
    system_prompt='You are the Judge agent, your responsibility is to evaluate the answer of another agent (B-Cell), '
                  'and adjust it\'s answer to ensure that it is sticking with the rules and standards provided.',
    instructions="""
                 Considering the **persona**, the **correctness** and **completeness** of the response, 
                 and **compliance** with the safety rules.
                 You have the same tools and information available to B-Cell agent. 
                 Please, define each criteria as True if the answer provided matches it.
                 You can define completeness as True if the missing information breaks some other criteria.
                 If necessary, write a corrected version of it.
                 """
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


async def interaction(query: str, dependencies: DialogContext,
                      chat_history, model=main_model, sec_model=None, passes=3):
    with Timer(initial_text='\nMain model', logger=logfire.info):
        query = (await transcriber.run(query)).output.english_transcription
        b_cell_result = await bcell.run(query, message_history=chat_history, deps=dependencies, model=model)

    if sec_model:
        with Timer(initial_text='\nJudge model', logger=logfire.info):
            criteria = False
            retries = 0
            while not criteria and retries<passes:
                judgement = (await judge.run(
                    [f'User query: {query}',
                     f'B-Cell answer: {b_cell_result.output.answer}',
                     f'Provided sources:{b_cell_result.output.sources}'],
                    deps=dependencies,
                    model=sec_model
                ))
                b_cell_result.output.answer = judgement.output.adjusted_answer
                b_cell_result.output.sources = judgement.output.sources
                criteria = judgement.output.criteria
                criteria = criteria.compliance and \
                           criteria.correctness
                retries+=1
    return b_cell_result

async def transcribe(audio: bytes, audio_type='audio/mp3') -> str:
    return (await transcriber.run([BinaryContent(audio, media_type=audio_type)])).output.english_transcription


async def tts(text:str) -> bytes:
    with Timer(initial_text='\nText-To-Speech', logger=logfire.info):
        completion = openai_client.chat.completions.create(
            model="gpt-audio-mini",
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

def interaction_sync(query, dependencies: DialogContext, chat_history, model, sec_model, passes):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(interaction(query, dependencies, chat_history, model, sec_model, passes))

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

    # ## Text loop
    while True:
        query_input = input('User: ')
        print()
        result = interaction_sync(query_input, deps, history, model=main_model, sec_model=judge_model, passes=3)
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
