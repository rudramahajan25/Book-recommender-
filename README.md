📚 Semantic Book Recommender
A semantic, content-based book recommendation system that understands what you want to read and how you want the book to feel. It uses modern text embeddings, a custom vector store, and tone-based filtering to deliver highly relevant book suggestions from natural-language queries.
🚀 Features
Semantic Search: Powered by nomic-ai/nomic-embed-text-v1.5 embeddings.
Content-Based Filtering: Matches books by meaning rather than ratings.
Custom Vector Database: Fast approximate nearest-neighbor (ANN) search.
Tone/Mood Signals: Optional emotion extraction adds “feel-based” filtering.
Interactive UI: Built with Gradio — type your query and browse the recommended books instantly.
📊 Dataset
Dataset: 7k Books with Metadata
Kaggle: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
Data processing includes:
Cleaning noisy categories
Handling missing values
Removing extremely short descriptions
Embedding-ready text preprocessing
🧠 Tech Stack
Python
Nomic Embeddings (nomic-embed-text-v1.5)
Custom Vector Store (ANN search)
Pandas / NumPy
Gradio
Optional sentiment & tone analysis
🖥️ Installation
git clone https://github.com/rudramahajan25/Book-recommender-.git
cd Book-recommender-
Create virtual environment (optional):
python3 -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
Install dependencies:
pip install -r requirements.txt
▶️ Run the App
python dashboard.py
This opens the Gradio UI in your browser.
📸 Media
Add your demo images like this:
Your folder structure:
media/
   dashboard_ui.png
   preview.png
   demo.gif
Example in README:
## 📸 Demo Preview
![Dashboard UI](media/dashboard_ui.png)

![Search Example](media/preview.png)
If you upload videos to GitHub releases or repo assets, use this:
https://github.com/rudramahajan25/Book-recommender-/assets/YOUR_ASSET_ID
🎯 Inspiration & Credits
This project was inspired by:
📌 “LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)”
By Aleksa Gordić (AIshaman)
YouTube: https://youtu.be/Q7mS1VHm3Yw
Extended in this repository with:
Nomic embeddings instead of OpenAI
A custom vector database
Mood/emotion filtering
Additional preprocessing and quality improvements
