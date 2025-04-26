import base64
import json
import os
import logging
from collections import deque
from datetime import datetime
import asyncio
from time import time

from dotenv import load_dotenv

from agent_tools import tools
from typing import Sequence, List, Iterator, Optional, Tuple
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import HumanMessage, trim_messages, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END

from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import InMemoryChatMessageHistory

from elevenlabs.client import ElevenLabs, AsyncElevenLabs
from elevenlabs import play

from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini-audio-preview",
    temperature=0.6,
    model_kwargs={
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "mp3"},
    }
)

dialog_logger = logging.getLogger(__name__)
claims = json.load(open("knowledge/talvey-claims.json", 'r'))

tooled_llm = llm.bind_tools(tools)

# elevanlabs_client = ElevenLabs()
elevanlabs_client = AsyncElevenLabs()
TARGET_VOICE_IDS = [
    os.getenv('ELEVEN_LABS_VOICE_ID_1'),
    os.getenv('ELEVEN_LABS_VOICE_ID_2'),
    os.getenv('ELEVEN_LABS_VOICE_ID_3'),
]
DEFAULT_MODEL_ID = "eleven_multilingual_sts_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", open('prompt.txt','r').read()),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=int(os.environ['CONTEXT_WINDOW_SIZE'])//5,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human",
)
runnable = prompt | trimmer | tooled_llm

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    context: List[Document]
    talvey_context: str

workflow = StateGraph(state_schema=State)

tool_executor = ToolNode(tools)
def should_continue(state: State) -> str:
    last_message = state['messages'][-1]
    # Call the tool if the last AI message asks for it
    if (isinstance(last_message, AIMessage)
            and hasattr(last_message, 'tool_calls')
            and last_message.tool_calls):
        return "call_tools"
    else:
        return END

# Define the function that calls the model
# and update message history with response
def call_model(state: State):
    response = runnable.invoke(state)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
workflow.add_node("call_tools", tool_executor)
workflow.add_conditional_edges(
    "model",
    should_continue,
    {
        "call_tools": "call_tools",
        END: END
    }
)
workflow.add_edge("call_tools", "model")

memory = MemorySaver()
chat_app = workflow.compile(checkpointer=memory)


# lang: Input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
def interaction(query, config, context,
                lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None, pprint=False):

    if not chat_history:
        chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(HumanMessage(query))

    input_dict = {
        "messages": list(chat_history.messages),
        "language": lang,
        "context": context,
        "talvey_context": claims
    }

    output = chat_app.invoke(input_dict, config)
    if not output["messages"][-1].content:
        output["messages"][-1].content = output["messages"][-1].additional_kwargs['audio']['transcript']
    chat_history.add_message(output["messages"][-1])

    if pprint:
        output["messages"][-1].pretty_print()  # output contains all messages in state
        print()
    return output

def text_interaction(query, config, context,
                lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None):
    return interaction(query, config, context, lang, chat_history)["messages"][-1].content

async def convert_voice(audio_bytes, voice_id):
    converted = elevanlabs_client.speech_to_speech.convert(
        audio=audio_bytes,
        voice_id=voice_id,  # Garanta que esta variável de ambiente está definida
        model_id="eleven_multilingual_sts_v2",  # Ou a variável DEFAULT_MODEL_ID se preferir
        output_format="mp3_44100_128",  # Ou a variável DEFAULT_OUTPUT_FORMAT
    )

    converted_audio_bytes = b""
    async for chunk in converted:
        if chunk:  # Verifica se o chunk não é None ou vazio
            converted_audio_bytes += chunk

    return converted_audio_bytes

async def gather_voices(encoded_audio):
    audio_bytes = base64.b64decode(encoded_audio)
    voice_1 = convert_voice(audio_bytes, os.getenv('ELEVEN_LABS_VOICE_ID_1'))
    voice_2 = convert_voice(audio_bytes, os.getenv('ELEVEN_LABS_VOICE_ID_2'))
    voice_3 = convert_voice(audio_bytes, os.getenv('ELEVEN_LABS_VOICE_ID_3'))

    voices = await asyncio.gather(voice_1, voice_2, voice_3)
    return voices


def mixed_interaction(query, config, context, lang='en',
                      chat_history: Optional[InMemoryChatMessageHistory]=None) -> Iterator[bytes]:
    encoded_audio = (interaction(query, config, context, lang, chat_history)["messages"][-1]
                        .additional_kwargs['audio']['data'])
    return asyncio.run(convert_voice(encoded_audio,  os.getenv('ELEVEN_LABS_VOICE_ID_1')))

async def audio_interaction(audio_path, config, context:deque,
                      lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None, speak=False) \
        -> Iterator[bytes]:

    with open(audio_path, "rb") as f:
        audio_data = f.read()
    audio_b64 = base64.b64encode(audio_data).decode()

    if not chat_history:
        chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(HumanMessage('', audio={"data": audio_b64}))

    input_dict = {
        "messages": list(chat_history.messages),
        "language": lang,
        "context": context,
        "talvey_context": claims
    }

    output = chat_app.invoke(input_dict, config)
    chat_history.add_message(output["messages"][-1])

    encoded_audio = output["messages"][-1].additional_kwargs['audio']['data']
    audio_bytes = base64.b64decode(encoded_audio)
    answer = await convert_voice(audio_bytes, os.getenv('ELEVEN_LABS_VOICE_ID'))
    if speak:
        play(answer)
        # play(audio_bytes)
    return answer

async def voices_interaction(audio_path, config, context:deque,
                      lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None, speak=False) \
        -> Tuple[Iterator[bytes]]:

    with open(audio_path, "rb") as f:
        audio_data = f.read()
    audio_b64 = base64.b64encode(audio_data).decode()

    if not chat_history:
        chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(HumanMessage('', audio={"data": audio_b64}))

    input_dict = {
        "messages": list(chat_history.messages),
        "language": lang,
        "context": context,
        "talvey_context": claims
    }

    output = chat_app.invoke(input_dict, config)
    chat_history.add_message(output["messages"][-1])

    encoded_audio = output["messages"][-1].additional_kwargs['audio']['data']
    voices = await gather_voices(encoded_audio)
    if speak:
        play(voices[0])
    return voices

async def async_main():
    # language = 'pt'
    language = 'en'
    thread_id = str(datetime.now())
    ctx = deque(maxlen=20)

    configuration = {"configurable": {"thread_id": thread_id}}
    # while True:
    #     query_input = input('User: ')
    #     print()
    #     interaction(query_input, configuration, ctx, language, pprint=True)
    await voices_interaction('/home/arthur/Documents/query.mp3', configuration, ctx, language, speak=True)
    # await audio_interaction('/home/arthur/Documents/query.mp3', configuration, ctx, language, speak=True)

if __name__ == '__main__':
    ts = time()
    asyncio.run(async_main())
    te = time() - ts
    print(te)