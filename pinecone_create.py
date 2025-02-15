import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = os.getenv("PINECONE_INDEX")

try:
    pc.delete_index(name=index_name)
except:
    pass

pc.create_index(
    name=index_name,
    dimension=3072, # dimensionalidade do text-embedding-3-large
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)