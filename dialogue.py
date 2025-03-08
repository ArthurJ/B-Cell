import json
import os
from collections import deque
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from typing import Sequence, List, Iterator, Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import HumanMessage, trim_messages, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from elevenlabs.client import ElevenLabs
from elevenlabs import play

from langchain_openai import ChatOpenAI
from openai import OpenAI
openai_client = OpenAI()

claims = json.load(open("knowledge/talvey-claims.json", 'r'))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=5000)
mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5000)

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.6, max_tokens=5000)
# mini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0, max_tokens=5000)

elevanlabs_client = ElevenLabs()

def vector_retrieve(query):
    retrieved_docs = vector_store.similarity_search(query, k=5)
    return retrieved_docs

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = PineconeVectorStore.from_existing_index(os.getenv('PINECONE_INDEX'), embeddings)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", open('prompt.txt','r').read()),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=int(os.environ['CONTEXT_WINDOW_SIZE'])//5,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o-mini"), #lambda x: len(x),
    include_system=True,
    start_on="human",
)

def lobotomy(msg: AIMessage):
    if '</think>' in msg.content:
        return {"role":"assistant", "content":msg.content.split('</think>')[1]}
    return {"role": "assistant", "content": msg.content}

runnable = prompt | trimmer | llm | lobotomy

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    context: List[Document]
    talvey_context: str

workflow = StateGraph(state_schema=State)

# Define the function that calls the model
# and update message history with response
def call_model(state: State):
    response = runnable.invoke(state)
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
chat_app = workflow.compile(checkpointer=memory)

def transcribe(audio_path, lang=None):
    if not lang:
        with open(audio_path, 'rb') as audio_file:
            return openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            ).text
    with open(audio_path, 'rb') as audio_file:
        return openai_client.audio.transcriptions.create(
            model="whisper-1",
            language=lang,
            file=audio_file
        ).text

def rewrite_query(query):
    return mini_llm.invoke(open('prompt_rewriter.txt', 'r').read() + f'{query}').content

def is_about_immunology(query):
    judgement = mini_llm.invoke('Just True or False: Does the following sentence '
                                'contains **relevant** content or question about immunology?'
                                f'\n{query}').content
    return judgement.split(':')[0].split('.')[0].split()[0] == 'True'

def text_to_speech(text_content):
    audio = elevanlabs_client.text_to_speech.convert(
        text=text_content,
        voice_id=os.getenv('ELEVEN_LABS_VOICE_ID'),
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    return audio

# lang: Input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
def text_interaction(query, config, context,
                     lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None, pprint=False):
    rewritten_query = rewrite_query(query)
    # if is_about_immunology(rewritten_query):
    #     context.append(vector_retrieve(rewritten_query))
    context.append(vector_retrieve(rewritten_query))

    if not chat_history:
        chat_history = InMemoryChatMessageHistory()
    chat_history.add_message(HumanMessage(query))

    input_dict = {
        "messages": list(chat_history.messages),
        "language": lang,
        "context": context,
        "talvey_context": claims
    }

    output= chat_app.invoke(input_dict, config)
    chat_history.add_message(output["messages"][-1])

    if pprint:
        output["messages"][-1].pretty_print()  # output contains all messages in state
        print()
    return output["messages"][-1].content

def mixed_interaction(query, config, context, lang='en',
                      chat_history: Optional[InMemoryChatMessageHistory]=None) -> Iterator[bytes]:
    text_content = text_interaction(query, config, context, lang, chat_history)
    return text_to_speech(text_content)

def audio_interaction(audio_path, config, context:deque,
                      lang='en', chat_history: Optional[InMemoryChatMessageHistory]=None, speak=False) \
        -> Iterator[bytes]:
    query = transcribe(audio_path, lang)
    text_content = text_interaction(query, config, context, lang, chat_history)
    audio = text_to_speech(text_content)
    if speak:
        play(audio)
    return audio


if __name__ == '__main__':

    # language = 'pt'
    language = 'en'
    thread_id = str(datetime.now())
    ctx = deque(maxlen=20)

    configuration = {"configurable": {"thread_id": thread_id}}
    while True:
        query_input = input('User: ')
        print()
        text_interaction(query_input, configuration, ctx, language, pprint=True)
