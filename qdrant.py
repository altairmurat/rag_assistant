import json
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────

FILE_PATH = "clean_data.txt"                  # your cleaned messages file
COLLECTION_NAME = "telegram_altf4"

# Good multilingual model (Russian + English + many others)
# You can also try: "intfloat/multilingual-e5-large" (1024 dim, very strong)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768 dim

# Qdrant connection
# client = QdrantClient(":memory:")                  # in-memory (fast for testing)
client = QdrantClient(path="./qdrant_telegram_db")   # persistent on disk
# client = QdrantClient(url="http://localhost:6333") # local docker/server


# ────────────────────────────────────────────────────────────────
# 1. Load cleaned texts
# ────────────────────────────────────────────────────────────────

def load_clean_texts(filepath: str) -> List[str]:
    messages = []
    current = []
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line == "":
                    if current:
                        messages.append("\n".join(current).strip())
                        current = []
                else:
                    current.append(line)
        
        if current:
            messages.append("\n".join(current).strip())
        
        # Remove empty
        messages = [msg for msg in messages if msg.strip()]
        
        print(f"Loaded {len(messages)} non-empty messages from {filepath}")
        return messages
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return []


clean_texts = load_clean_texts(FILE_PATH)

if not clean_texts:
    print("No messages loaded → exiting")
    exit(1)


# ────────────────────────────────────────────────────────────────
# 2. Prepare embedding model
# ────────────────────────────────────────────────────────────────

print(f"Loading embedding model: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)
VECTOR_DIM = embedder.get_sentence_embedding_dimension()
print(f"Embedding dimension: {VECTOR_DIM}")


# ────────────────────────────────────────────────────────────────
# 3. (Re)create collection
# ────────────────────────────────────────────────────────────────

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted old collection '{COLLECTION_NAME}'")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_DIM,
        distance=Distance.COSINE
    )
)
print(f"Created collection '{COLLECTION_NAME}'")


# ────────────────────────────────────────────────────────────────
# 4. Embed + upload
# ────────────────────────────────────────────────────────────────

print("Encoding messages...")
embeddings = embedder.encode(
    clean_texts,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()

print("Uploading to Qdrant...")

points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "text": text,
            "source": "altf4_telegram_channel",
            # You can add more fields if you kept original metadata:
            # "date": "...",
            # "message_id": ...,
        }
    )
    for embedding, text in zip(embeddings, clean_texts)
]

client.upload_points(
    collection_name=COLLECTION_NAME,
    points=points,
    batch_size=128,          # adjust based on RAM
    wait=True                # wait until upload finishes
)

print(f"Successfully uploaded {len(points)} messages")


# ────────────────────────────────────────────────────────────────
# 5. Quick test query
# ────────────────────────────────────────────────────────────────

test_query = "кто владелец канала alt f4"   # change to any question you want

print(f"\nTest query: {test_query}")

query_embedding = embedder.encode(test_query).tolist()

search_result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=5
)

print("\nTop results:")
for i, point in enumerate(search_result.points, 1):
    score = point.score
    text_preview = point.payload.get("text", "???")[:180].replace("\n", " ")
    print(f"{i}. [{score:.4f}] {text_preview}...")


print("\nDone.")