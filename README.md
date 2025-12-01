📚 Semantic Book Recommender
A semantic, content-based book recommendation system that understands what you want to read and how you want the book to feel. It uses modern text embeddings, a custom vector store, and tone-based filtering to deliver highly relevant book suggestions from natural-language queries.
🚀 Features
Semantic Search: Uses nomic-ai/nomic-embed-text-v1.5 to embed 7k+ book descriptions.
Content-Based Filtering: Matches books by meaning, not ratings.
Custom Vector Database: Fast nearest-neighbor similarity search in milliseconds.
Tone/Mood Signals: Optional emotion analysis to refine recommendations by feel.
Interactive UI: Gradio dashboard for typing queries and browsing recommended books with covers + metadata.
📊 Dataset
Dataset: 7k Books with Metadata
Kaggle Link: https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata
Data processing includes:
Cleaning noisy categories
Handling missing values
Removing extremely short descriptions
Preparing text for embedding
🧠 Tech Stack
Python
Nomic Embeddings (nomic-embed-text-v1.5)
Custom Vector Store (ANN search)
Pandas / NumPy
Gradio (front end)
Optional sentiment/tone extraction
🖥️ Installation & Setup
# Clone the repository
git clone https://github.com/yourusername/semantic-book-recommender.git
cd semantic-book-recommender

# Create environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
▶️ Running the App
python dashboard.py
This will launch the Gradio UI in your browser.
📸 Media
You can add images, GIFs, or demonstration videos here.
Replace the placeholders below with your own media links:
![Demo Screenshot](media/Demo.mov)

![UI Preview](media/Dashboard UI.png)

https://github.com/yourusername/semantic-book-recommender/assets/your-video-id
🎯 Inspiration & Credits
This project was inspired by the YouTube tutorial:
📌 “LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)”
By Aleksa Gordić (AIshaman)
Link: https://youtu.be/Q7mS1VHm3Yw
I extended the project by:
Using Nomic embeddings instead of OpenAI
Implementing a custom vector store
Adding tone/emotion filtering for more expressive recommendations
