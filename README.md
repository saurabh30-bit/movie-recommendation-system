# 🎬 Movie Recommendation System

A Content-Based Movie Recommendation Engine built with Python, Pandas, and Scikit-Learn.

## 🚀 Features
- **Content-Based Filtering**: Recommends movies similar to a user's input based on metadata (genres, keywords, cast).
- **TF-IDF Vectorization**: Converts raw text data into meaningful numerical features.
- **Cosine Similarity**: Mathematically calculates the distance between movie vectors to find the most relevant matches.
- **Extensible**: Designed as an object-oriented class that can be easily connected to a web frontend or REST API.

## 📦 Tech Stack
- **Python 3.9+**
- **Pandas**: Data manipulation and analysis
- **Scikit-Learn**: Machine learning utilities (TF-IDF, Cosine Similarity)

## 🛠️ Local Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saurabh30-bit/movie-recommendation-system.git
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the recommendation script:
   ```bash
   python recommender.py
   ```

## Configuration
Copy .env.example to .env and add your API keys if you plan to fetch live data.

