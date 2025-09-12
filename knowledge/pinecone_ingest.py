from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import glob
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

text_splitter = SemanticChunker(embeddings,
                                breakpoint_threshold_type="gradient")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX'))
vector_store = PineconeVectorStore(embedding=embeddings, index=index)

pages = []
md_paths = glob.glob("Summaries/**/*.md", recursive=True)
for file_path in md_paths:
    if Path(file_path).is_file():
        print(file_path)
        loader = UnstructuredMarkdownLoader(file_path)
        for page in loader.lazy_load():
            pages.append(page)

all_splits = text_splitter.split_documents(pages)
print(f"Split blog post into {len(all_splits)} sub-documents.")
vector_store.add_documents(documents=all_splits)


pages = []
pdf_paths = glob.glob("Originals/**/*.pdf", recursive=True)
for file_path in pdf_paths:
    if Path(file_path).is_file():
        print(file_path)
        loader = PyPDFLoader(file_path)
        for page in loader.lazy_load():
            pages.append(page)

all_splits = text_splitter.split_documents(pages)
print(f"Split blog post into {len(all_splits)} sub-documents.")
vector_store.add_documents(documents=all_splits)

# if __name__=='__main__':
#
#     while True:
#         query = input('Ask:')
#         docs = vector_store.similarity_search(query, k=3)
#         for doc in docs:
#             print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')