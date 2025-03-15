# YouTube Transcript Summarizer

## Overview
The **YouTube Transcript Summarizer** is a web-based application built with **Streamlit** that extracts transcripts from YouTube videos, summarizes them, and provides keyword extraction, topic modeling, and sentiment analysis. The project also includes a Chrome extension and a Flask backend to enhance usability.

## Features
- 🎥 Extracts transcripts from YouTube videos.
- ✂️ Summarizes long transcripts into concise text.
- 🔑 Extracts important keywords.
- 📊 Performs topic modeling using LDA.
- 😊 Conducts sentiment analysis to determine the video's tone.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** Flask
- **Libraries Used:**
  - `youtube_transcript_api` (Transcript retrieval)
  - `transformers` (Summarization model)
  - `nltk` (Text processing)
  - `scikit-learn` (Topic modeling)
  - `textblob` (Sentiment analysis)

## Installation & Setup

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Streamlit App
```sh
streamlit run app.py
```

## Usage
1. Enter the **YouTube video URL**.
2. Click **Summarize** to process the transcript.
3. View the summary, keywords, topics, and sentiment analysis.

## Possible Errors & Fixes
- **Missing `punkt_tab` in NLTK:**
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('wordnet')
  nltk.download('stopwords')
  nltk.download('omw-1.4')
  ```
- **Transcript not available:** Ensure the video has subtitles enabled.

## Future Enhancements
- 🛠️ Add multilingual support.
- 🌎 Improve summarization accuracy with better models.
- 📌 Enable caching for faster processing.

## Author
Developed by **Monpara Romil Kamleshbhai** 🚀