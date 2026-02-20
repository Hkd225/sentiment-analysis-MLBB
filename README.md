# Sentiment Analysis of Mobile Legends: Bang Bang Indonesian Reviews
## Machine Learning vs Deep Learning Comparison
### By Muhammad Auffa Hakim Aditya

This project presents a complete Natural Language Processing (NLP) pipeline for Sentiment Analysis of Indonesian user reviews of Mobile Legends: Bang Bang collected from the Google Play Store.

The project was developed by Muhammad Auffa Hakim Aditya to compare the performance of Machine Learning and Deep Learning models in classifying player sentiment (feedback, complaints, and praises) in Indonesian text data.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Scrape 40,000 Indonesian reviews from Google Play Store.
2. Perform text preprocessing specialized for Indonesian gaming slang.
3. Apply lexicon-based sentiment labeling (Positive, Negative, Neutral).
4. Compare performance between:
   - GRU (Deep Learning)
   - Word2Vec + LSTM (Deep Learning)
   - TF-IDF + SVM (Machine Learning)
5. Determine the best-performing model to identify player satisfaction.

------------------------------------------------------------

DATASET INFORMATION

Source         : Google Play Store
Application    : Mobile Legends: Bang Bang (MLBB)
Language       : Indonesian
Total Reviews  : ~40,000

Scraping library used:
- google-play-scraper

------------------------------------------------------------

NLP PIPELINE

1. Data Cleaning
   - Remove URLs, mentions (@), and hashtags (#)
   - Remove numbers, punctuation, and emojis

2. Case Folding
   - Convert all text to lowercase

3. Slang Word Normalization
   - Custom Indonesian slang dictionary (e.g., "yg" -> "yang")
   - MLBB specific terminology normalization:
     * "mabar" -> "main bareng"
     * "ngeleg" -> "lag" / "gangguan koneksi"
     * "buta map" -> "kurang kesadaran peta"
     * "bacot" -> "toxic"
     * "dark system" -> "matchmaking buruk"

4. Tokenization
   - Using NLTK for word splitting

5. Stopword Removal
   - NLTK Indonesian & English stopwords
   - Custom stopwords (removing non-informative common gaming words)

6. Stemming
   - Using Sastrawi Stemmer for Indonesian root word extraction

------------------------------------------------------------

SENTIMENT LABELING

Sentiment labeling is performed using a Lexicon-Based approach:

- Positive lexicon: "seru", "mantap", "mulus", "skin gratis"
- Negative lexicon: "lag", "sinyal merah", "toxic", "afk", "bocil"
- Negation handling: "tidak lag", "nggak seru", etc.

Sentiment classes:
- Positive (Satisfied players)
- Negative (Complaints/Bug reports/Toxic environment)
- Neutral (General feedback/Inquiries)

------------------------------------------------------------

MODEL EXPERIMENTS

SCHEMA 1 — GRU (Deep Learning)
- Tokenizer + Padding
- Embedding Layer
- Gated Recurrent Unit (GRU) Layer
- Dense + Dropout
- Softmax output (3 classes)

SCHEMA 2 — Word2Vec + LSTM (Deep Learning)
- Gensim Word2Vec training (Contextual word relations)
- Embedding Matrix initialization
- Long Short-Term Memory (LSTM) Layer
- EarlyStopping to prevent overfitting
- Dense + Dropout

SCHEMA 3 — TF-IDF + SVM (Machine Learning)
- TfidfVectorizer (Word importance weighting)
- Linear Support Vector Machine (SVM)
- Traditional ML classification pipeline

------------------------------------------------------------

MODEL EVALUATION

Models are evaluated using:
- Accuracy Score
- Precision, Recall, and F1-Score
- Comparison between traditional ML vs. Sequential DL approaches

The best model is selected based on the highest accuracy in predicting gaming-specific sentiments.

------------------------------------------------------------

MODEL SAVING

All trained models are saved for deployment:

- .keras (Deep Learning models - GRU/LSTM)
- .joblib (SVM and Tokenizers)
- .model (Word2Vec Embeddings)

------------------------------------------------------------

TESTING WITH NEW SENTENCES

Example: "Sinyal sering merah pas war padahal kuota banyak."
Pipeline: Preprocessing -> Vectorization -> Model Prediction -> Result: NEGATIVE.

------------------------------------------------------------

INSTALLATION

Install dependencies:
pip install -r requirements.txt

------------------------------------------------------------

HOW TO RUN

1. Clone repository:
   git clone https://github.com/YOUR_USERNAME/sentiment-analysis-mlbb-indonesia.git

2. Install requirements
3. Run the Python script or Jupyter Notebook

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

Project focus:
- Natural Language Processing (NLP)
- Sentiment Analysis (Indonesian Gaming Community)
- Machine Learning vs Deep Learning
- Word Embedding (Word2Vec)
- Sequence Modeling (GRU & LSTM)

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Mobile Legends Sentiment Analysis
- Indonesian NLP Project
- MLBB Review Classification
- Machine Learning vs Deep Learning
- TF-IDF SVM
- Word2Vec LSTM
- Text Mining Gaming
- NLP Portfolio Project
