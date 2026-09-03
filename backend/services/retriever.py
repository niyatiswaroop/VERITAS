import os
import chromadb
from sentence_transformers import SentenceTransformer

from models import NO_EVIDENCE_MESSAGE

# Max L2 distance for a chunk to count as relevant evidence. Chunks farther than
# this are treated as "not about the claim" so the claim is reported Unsupported
# rather than being force-matched to whatever is closest in the knowledge base.
# Calibrated on the current KB: real matches land ~0.4-0.5, off-topic queries ~1.6+.
MAX_EVIDENCE_DISTANCE = 1.0

# Custom chunking function
def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 30:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

class EvidenceRetriever:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
        # Initialize models just like in the internship
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        
        # Get or create collection
        try:
            self.client.delete_collection("veritas_knowledge_base")
        except:
            pass
        self.collection = self.client.get_or_create_collection("veritas_knowledge_base")
        
        self.has_documents = False
        self._load_and_index_documents()

    def _load_and_index_documents(self):
        """Loads documents from the data directory, chunks them, and adds to ChromaDB."""
        current_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(current_dir, self.data_dir)
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        chunks = []
        ids = []
        embeddings = []
        
        doc_id = 0
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                    # Use the internship chunking logic
                    file_chunks = split_into_chunks(text, chunk_size=200, overlap=20) # Slightly smaller chunk size for claims
                    
                    for chunk in file_chunks:
                        embedding = self.embedder.encode(chunk).tolist()
                        chunks.append(chunk)
                        embeddings.append(embedding)
                        ids.append(f"chunk_{doc_id}")
                        doc_id += 1
                            
        if chunks:
            # Add to ChromaDB exactly like the internship
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids
            )
            self.has_documents = True
            print(f"Loaded {len(chunks)} chunks into ChromaDB.")
        else:
            print("No documents found in knowledge base.")

    def retrieve(self, query: str, top_k: int = 2,
                 max_distance: float = MAX_EVIDENCE_DISTANCE) -> str:
        """Retrieves the top-k most relevant evidence chunks for a query from ChromaDB.

        Returns NO_EVIDENCE_MESSAGE when the knowledge base has nothing close
        enough to the query, so a claim with no coverage is reported as
        Unsupported instead of being matched to an unrelated chunk.
        """
        if not self.has_documents:
            return NO_EVIDENCE_MESSAGE

        # Encode query like in internship
        query_embedding = self.embedder.encode(query).tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_documents = results['documents'][0]
        distances = results['distances'][0]

        # Nothing retrieved, or the closest chunk is still too far to be
        # considered evidence for this claim.
        if not retrieved_documents or distances[0] > max_distance:
            return NO_EVIDENCE_MESSAGE

        return "\n\n".join(retrieved_documents)

retriever = EvidenceRetriever()
