import os
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ------------------------------------------------------------
#  Enable MPS acceleration on Apple Silicon
# ------------------------------------------------------------
if torch.backends.mps.is_available():
    torch.set_default_device("mps")
    print("🔥 Using Apple MPS GPU")
else:
    print("⚠️ MPS not available, using CPU")

# ------------------------------------------------------------
#  Load environment
# ------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------
#  Load dataset
# ------------------------------------------------------------
books = pd.read_csv("books_with_emotions.csv")

# Replace missing thumbnails with a placeholder BEFORE appending &fife
books["thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg")
books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"

# ------------------------------------------------------------
#  Load descriptions for vector search
# ------------------------------------------------------------
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

raw_documents = TextLoader("tagged_description.txt").load()

# Split into smaller chunks safely for embeddings
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=512,    # safe chunk size for Nomic embeddings
    chunk_overlap=50
)
documents = text_splitter.split_documents(raw_documents)

# ------------------------------------------------------------
#  BEST LOCAL EMBEDDINGS — Nomic v1.5
# ------------------------------------------------------------
from transformers import AutoTokenizer, AutoModel
import numpy as np

class BetterEmbeddings:
    def __init__(self):
        model_name = "nomic-ai/nomic-embed-text-v1.5"

        print("⏳ Loading Nomic model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        print("✅ Loaded nomic-embed-text-v1.5")

    def _embed(self, texts, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]

        embeddings_list = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
                self.model.to("mps")

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeds = outputs.last_hidden_state[:, 0]  # CLS embedding
                batch_embeds = torch.nn.functional.normalize(batch_embeds, p=2, dim=1)
                embeddings_list.append(batch_embeds.cpu().numpy())

        return np.vstack(embeddings_list)

    def embed_documents(self, docs):
        return self._embed(docs, batch_size=16).tolist()

    def embed_query(self, query):
        return self._embed([query], batch_size=1)[0].tolist()


embedding_model = BetterEmbeddings()

# ------------------------------------------------------------
#  Chroma DB creation/loading
# ------------------------------------------------------------
from langchain_chroma import Chroma

persist_dir = "chroma_nomic"

def check_and_load_db():
    """
    Load existing DB if valid, otherwise create new one.
    """
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            db = Chroma(persist_directory=persist_dir)
            # Test a simple query to verify dimension
            db.similarity_search("test", k=1)
            print("⚡ Loaded existing Chroma DB (no rebuild needed)")
            return db
        except Exception as e:
            print(f"⚠️ Existing DB invalid ({e.__class__.__name__}). Recreating...")
            import shutil
            shutil.rmtree(persist_dir)

    # Create new DB
    print("⚠️ Creating new Chroma DB with Nomic embeddings...")
    db = Chroma.from_documents(
        documents,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print("✅ New Chroma DB created")
    return db

db_books = check_and_load_db()

# ------------------------------------------------------------
#  Retrieval logic with improved filtering
# ------------------------------------------------------------
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
):
    results = db_books.similarity_search_with_relevance_scores(query, k=initial_top_k)
    results = [r for r in results if r[1] > 0.30]

    # Extract ISBNs
    book_ids = [int(r[0].page_content.strip('"').split()[0]) for r in results]

    # Match with dataset
    book_recs = books[books["isbn13"].isin(book_ids)].head(initial_top_k)

    # Category filtering
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Tone re-ranking
    if tone == "Happy":
        book_recs = book_recs.sort_values("joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values("surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values("anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values("fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values("sadness", ascending=False)

    return book_recs.head(final_top_k)

# ------------------------------------------------------------
#  Format results for Gradio
# ------------------------------------------------------------
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        desc = " ".join(row["description"].split()[:30]) + "..."
        authors = row["authors"].split(";")
        if len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            authors_str = f"{', '.join(authors[:-1])}, and {authors[-1]}"
        else:
            authors_str = authors[0]

        caption = f"{row['title']} by {authors_str}: {desc}"
        results.append((row["large_thumbnail"], caption))

    return results

# ------------------------------------------------------------
#  Gradio UI
# ------------------------------------------------------------
import gradio as gr

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender ")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book you're interested in:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Tone", value="All")
        submit_button = gr.Button("🔍 Find Recommendations")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label="Books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
