import os
import pinecone
import uuid
from datetime import datetime
from dotenv import load_dotenv

from commands.embed import get_embedding

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1081"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1081"

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


class MemoryRetrievalBrain:
    def __init__(self, table, dimension=1536, metric="cosine", pod_type="p1"):
        self.table_name = table
        self.dimension = dimension
        self.metric = metric
        self.pod_type = pod_type
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name,
                dimension=dimension,
                metric=metric,
                pod_type=pod_type
            )
        self.index = pinecone.Index(self.table_name)

    def memorize(self, text, id: str = None):
        embedding = get_embedding(text)
        if id is None:
            id = str(uuid.uuid4())
        today = datetime.today()
        self.index.upsert([
            (id, embedding, {"text": text, "year": today.year, "month": today.month})
        ])

    def remember(self, query, top_k=5):
        query_embedding = get_embedding(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )["matches"]
        memories = [m['metadata']['text'] for m in results]

        return memories

    def remember_with_ids(self, ids):
        results = self.index.fetch(ids)['vectors']
        memories = []
        for id in ids:
            if id in results:
                memories.append(results[id]['metadata']['text'])
            else:
                memories.append(None)
        return memories
