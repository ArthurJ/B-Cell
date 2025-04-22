import json
import logging
import os
from typing import List

from langchain_core.tools import tool
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

tool_logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vector_store = PineconeVectorStore.from_existing_index(os.getenv('PINECONE_INDEX'), embeddings)

@tool
def knowledge_retrieve(query:str) -> List[Document]:
    """Retrieve knowledge from scientific papers based on the query"""
    tool_logger.info(f'Knowledge access: {query}.')
    retrieved_docs = vector_store.similarity_search(query, k=5)
    return retrieved_docs

tools = [knowledge_retrieve]