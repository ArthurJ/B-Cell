import os
from collections import deque

from dotenv import load_dotenv
load_dotenv()

from typing import Sequence, List, Dict, Any, Iterator
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import HumanMessage, trim_messages, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from datetime import datetime

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from elevenlabs.client import ElevenLabs
from elevenlabs import play

from fastapi import UploadFile

from openai import OpenAI
openai_client = OpenAI()

# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="deepseek-r1:32b", temperature=0.3, keep_alive=int(os.getenv('KEEP_ALIVE')), max_tokens=5000)
# mini_llm = ChatOllama(model="aya-expanse:32b", temperature=0, max_tokens=5000)
# mini_llm = ChatOllama(model="phi4:latest", temperature=0, max_tokens=5000)

# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0.6, max_tokens=5000)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.6, max_tokens=5000)
mini_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5000)

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05", temperature=0.6, max_tokens=5000)
# mini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0, max_tokens=5000)

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
graph = Neo4jGraph(url=os.getenv('NEO4J_URI'),
                   username=os.getenv('NEO4J_USER'),
                   password=os.getenv('NEO4J_PASS'))

elevanlabs_client = ElevenLabs()

graph_reader = GraphCypherQAChain.from_llm(
    mini_llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

def vector_retrieve(query):
    retrieved_docs = vector_store.similarity_search(query, k=1)
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
    max_tokens=int(os.environ['CONTEXT_WINDOW_SIZE']),
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

def rewrite_query(query):
    return mini_llm.invoke(open('prompt_rewriter.txt', 'r').read() + f'{query}').content

def is_about_immunology(query):
    judgement = mini_llm.invoke('Just True or False: Does the following sentence '
                                'contains **relevant** content or question about immunology?'
                                f'\n{query}').content
    return judgement.split(':')[0].split('.')[0].split()[0] == 'True'

def text_interaction(query, config, context, lang='English', pprint=False):
    rewritten_query = rewrite_query(query)
    if is_about_immunology(rewritten_query):
        context.append(vector_retrieve(rewritten_query))

    input_dict = {
        "messages": [HumanMessage(query)],
        "language": lang,
        "context": context
    }

    output= chat_app.invoke(input_dict, config)
    if pprint:
        output["messages"][-1].pretty_print()  # output contains all messages in state
        print()
    return output["messages"][-1].content

def transcribe(audio_path):
    with open(audio_path, 'rb') as audio_file:
        return openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        ).text

def audio_interaction(audio_path, config, context, lang='English', speak=False) -> Iterator[bytes]:
    query = transcribe(audio_path)

    rewritten_query = rewrite_query(query)
    if is_about_immunology(rewritten_query):
        context.append(vector_retrieve(rewritten_query))

    input_dict = {
        "messages": [HumanMessage(query)],
        "language": lang,
        "context": context
    }

    output = chat_app.invoke(input_dict, config)
    # output["messages"][-1].pretty_print()

    audio = elevanlabs_client.text_to_speech.convert(
        text=output["messages"][-1].content,
        voice_id=os.getenv('ELEVEN_LABS_VOICE_ID'),
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    if speak:
        play(audio)
    return audio

def text_stream(query, config, context, lang='English', pprint=False):
    rewritten_query = rewrite_query(query)
    if is_about_immunology(rewritten_query):
        context.append(vector_retrieve(rewritten_query))

    input_dict = {
        "messages": [HumanMessage(query)],
        "language": lang,
        "context": context
    }

    for chunk in chat_app.stream(input_dict, config):
        yield chunk['model']['messages']['content']

def audio_stream(audio_path, config, context, lang='English'):
    query = transcribe(audio_path)

    rewritten_query = rewrite_query(query)
    if is_about_immunology(rewritten_query):
        context.append(vector_retrieve(rewritten_query))

    input_dict = {
        "messages": [HumanMessage(query)],
        "language": lang,
        "context": context
    }

    output = chat_app.invoke(input_dict, config)

    audio = elevanlabs_client.text_to_speech.convert_as_stream(
        text=output["messages"][-1].content,
        voice_id=os.getenv('ELEVEN_LABS_VOICE_ID'),
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    for chunk in audio:
        if isinstance(chunk, bytes):
            yield chunk

if __name__ == '__main__':

    # language = 'Português Brasileiro'
    language = 'English'
    thread_id = str(datetime.now())
    ctx = deque(maxlen=3)

    configuration = {"configurable": {"thread_id": thread_id}}
    while True:
        query_input = input('User: ')
        print()
        for chunk in text_stream(query_input, configuration, ctx, language):
            print(chunk, end=' ')