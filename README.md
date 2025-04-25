
# 🎬 Enhanced YouTube Transcript Summarizer with Sentiment Analysis

## 🧠 Overview
This project is a powerful Streamlit-based web application that extracts and processes **YouTube video transcripts** using advanced **Natural Language Processing (NLP)** techniques.

It provides:
- ✂️ **Summarization** using Transformer models (T5)
- 🔑 **Keyword Extraction** with TF-IDF & fallback lemmatization
- 🧠 **Topic Modeling** using LDA
- 😊 **Sentiment Analysis** using both VADER and TextBlob
- 📥 **Export** options to download the results as `.txt` or `.csv`

---

## 👨‍💻 Developer Info

- **Name:** Romil Monpara

---

## 🚀 Features

- 🔗 Input any YouTube video URL
- 📝 Extracts and cleans transcript using `youtube_transcript_api`
- 🤖 Summarizes content via Hugging Face T5 Transformer model
- 🧹 Keyword extraction using TF-IDF with fallback to word frequency
- 📚 Topic modeling via LDA (`scikit-learn`)
- ❤️ Sentiment analysis using:
  - `TextBlob` (Polarity, Subjectivity)
  - `NLTK`'s VADER (Positive, Negative, Neutral, Compound)
- 📊 Visual representation of sentiment results
- 📤 Downloadable results (TXT for summary, CSV for full data)

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Core Libraries:**
  - `youtube_transcript_api`, `pytube` – YouTube integration
  - `transformers` – T5 summarization model
  - `nltk`, `textblob` – NLP, sentiment analysis
  - `scikit-learn` – Topic modeling (LDA)
  - `matplotlib`, `pandas`, `base64` – Visuals & export
  - `streamlit` – UI & interaction

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/romilmonpara/youtube-transcript-streamlit-ui.git
```

### 2. Install Required Libraries
Make sure you have Python 3.7+ installed.

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Resources (First-time setup)
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🧪 How to Use

1. Open the app in your browser after launching Streamlit.
2. Paste a **YouTube video URL** in the input box.
3. Click **"Analyze Video"**.
4. View:
   - Video metadata (title, author, views, etc.)
   - Raw transcript (optional)
   - Summary
   - Keywords
   - Topics
   - Sentiment plots
5. Download:
   - 📄 `summary.txt`
   - 📊 `analysis.csv`

---

## 📌 Output Example

- **Summary:**
  > "In this video, the speaker discusses..."

- **Top Keywords:**
  > data science, machine learning, deep learning...

- **Topics:**
  > Topic 1: ai, data, learning  
  > Topic 2: video, algorithm, streamlit

- **Sentiment:**
  > Polarity: 0.15 | Subjectivity: 0.45  
  > VADER Compound Score: 0.74

---

## 🛠️ Error Handling

- **Transcript Not Available:**  
  > The video must have closed captions enabled.

- **Invalid URL:**  
  > Only standard YouTube links are accepted.

- **Model Error / CUDA Out of Memory:**  
  > Reduce summary length or input shorter videos.

---

## 🌟 Future Enhancements

- 🌐 Multilingual transcript support  
- ⚡ Faster summarization with GPU model serving  
- 🧠 Use more advanced models like BART or Pegasus  
- 🖼️ Improve UI with themes and mobile responsiveness  

---

## 📜 License

This project is for educational purposes only.

---

## 🙌 Acknowledgements

- Hugging Face Transformers  
- NLTK & TextBlob teams  
- Streamlit Community  
- YouTube Transcript API developers

---

## Author
- Developed by **Romil Monpara** 🚀
- **GitHub:** [@romilmonpara](https://github.com/romilmonpara)


---
