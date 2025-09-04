# Title: Enhanced YouTube Transcript Summarizer with Sentiment Analysis
# Name: Monpara Romil Kamleshbhai
# Enroll No.: 23002170210064

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pytube import YouTube
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
from collections import Counter
from langdetect import detect, LangDetectException
import re
import nltk
import base64
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 4000  # For summarization
MIN_TEXT_LENGTH = 50    # Minimum for processing
DEFAULT_SUMMARY_LENGTH = 500

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

@st.cache_resource
def load_models():
    """Load all ML models once and cache them"""
    try:
        # Use PyTorch backend instead of TensorFlow
        summarizer = pipeline(
            "summarization", 
            model="t5-small", 
            framework="pt",
            tokenizer="t5-small",
            torch_dtype='auto'
        )
        return {
            'summarizer': summarizer,
            'sentiment_analyzer': SentimentIntensityAnalyzer()
        }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Fallback to simple summarization without transformers
        return {
            'summarizer': None,
            'sentiment_analyzer': SentimentIntensityAnalyzer()
        }

def summarize_text(text: str, summarizer, max_length: int = DEFAULT_SUMMARY_LENGTH) -> str:
    """
    Summarizes text using T5 model or fallback method
    
    Args:
        text (str): Input text to summarize
        summarizer: Loaded summarization pipeline or None
        max_length (int): Maximum length of summary
        
    Returns:
        str: Generated summary or error message
    """
    if len(text) < MIN_TEXT_LENGTH:
        return "Text is too short to summarize."

    text = text[:MAX_TEXT_LENGTH]  # Limit for summarization model

    try:
        if summarizer is not None:
            # Use transformers model
            summary = summarizer(
                text, 
                max_length=min(len(text)//2, max_length), 
                min_length=30, 
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']
        else:
            # Fallback to extractive summarization
            return extractive_summarization(text)
    except Exception as e:
        logger.warning(f"Summarization failed, using fallback: {e}")
        return extractive_summarization(text)

def extractive_summarization(text: str, sentences_count: int = 3) -> str:
    """
    Fallback extractive summarization using TF-IDF
    
    Args:
        text (str): Input text to summarize
        sentences_count (int): Number of sentences to include in summary
        
    Returns:
        str: Extracted summary
    """
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= sentences_count:
            return text
        
        # Simple TF-IDF based sentence ranking
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        sentence_vectors = vectorizer.fit_transform(sentences)
        sentence_scores = sentence_vectors.sum(axis=1)
        
        top_sentences = sorted(
            enumerate(sentence_scores.flatten().tolist()),
            key=lambda x: x[1], 
            reverse=True
        )[:sentences_count]
        
        return ' '.join(sentences[i] for i, _ in sorted(top_sentences))
    except Exception as e:
        logger.error(f"Extractive summarization failed: {e}")
        # Final fallback - return first few sentences
        sentences = sent_tokenize(text)
        return ' '.join(sentences[:min(3, len(sentences))])

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """
    Split long text into chunks for processing
    
    Args:
        text (str): Input text to chunk
        max_chunk_size (int): Maximum size of each chunk
        
    Returns:
        List[str]: List of text chunks
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        if current_size + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords using TF-IDF with n-grams
    
    Args:
        text (str): Input text
        
    Returns:
        List[str]: Top keywords
    """
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=50)
        X = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        return [features[i] for i in scores.argsort()[-5:][::-1]]
    except Exception as e:
        logger.warning(f"TF-IDF keyword extraction failed: {e}")
        # Fallback to simple method if TF-IDF fails
        lemmatizer = WordNetLemmatizer()
        words = re.findall(r'\b\w+\b', text.lower())
        words = [lemmatizer.lemmatize(word) for word in words 
                if word not in stopwords.words('english') and len(word) > 2]
        return [word for word, _ in Counter(words).most_common(5)]

def topic_modeling(text: str) -> List[List[str]]:
    """
    Perform topic modeling on text using LDA
    
    Args:
        text (str): Input text
        
    Returns:
        List[List[str]]: Detected topics with top words
    """
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return [["Not enough text for topic modeling"]]

    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=1, stop_words='english', max_features=100)
        tf = vectorizer.fit_transform(sentences)

        if tf.shape[1] == 0:
            return [["Not enough unique words for topic modeling"]]

        lda_model = LatentDirichletAllocation(
            n_components=min(3, len(sentences)), 
            max_iter=5, 
            learning_method='online', 
            random_state=42
        )
        lda_model.fit(tf)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])

        return topics
    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        return [[f"Topic modeling failed: {str(e)}"]]

def detect_language(text: str) -> str:
    """
    Detect the language of the text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Detected language code
    """
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English

def check_transcript_quality(transcript) -> float:
    """
    Check the quality of the transcript
    
    Args:
        transcript: YouTube transcript data
        
    Returns:
        float: Quality score (words per second)
    """
    try:
        total_duration = sum([entry['duration'] for entry in transcript])
        total_text = sum([len(entry['text']) for entry in transcript])
        
        # Calculate words per second as a quality metric
        words_per_second = total_text / total_duration / 5  # Approximate word length
        
        return words_per_second  # Values > 2.5 indicate good quality
    except:
        return 0.0

def extract_video_id(url: str) -> Union[str, None]:
    """
    Extract video ID from various YouTube URL formats
    
    Args:
        url (str): YouTube URL
        
    Returns:
        str or None: Extracted video ID or None if invalid
    """
    patterns = [
        r'v=([^&]+)',
        r'youtu.be/([^?]+)',
        r'youtube.com/embed/([^?]+)',
        r'youtube.com/shorts/([^?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@st.cache_data(ttl=3600, show_spinner="Fetching transcript...")
def get_cached_transcript(video_id: str) -> Union[List[Dict], None]:
    """
    Get transcript with caching
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        List[Dict] or None: Transcript data or None if not available
    """
    try:
        return YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        return None

def get_video_metadata(video_id: str) -> Dict[str, Union[str, int]]:
    """
    Get basic metadata about the YouTube video
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        Dict: Video metadata (title, author, etc.)
    """
    try:
        yt = YouTube(f"https://youtu.be/{video_id}")
        return {
            'title': yt.title,
            'author': yt.author,
            'length': f"{yt.length // 60}:{yt.length % 60:02d}",
            'views': f"{yt.views:,}",
            'publish_date': yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else "Unknown"
        }
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}")
        return {}

def create_download_link(text: str, filename: str, label: str) -> str:
    """
    Create a downloadable link for text content
    
    Args:
        text (str): Content to download
        filename (str): Name for downloaded file
        label (str): Display text for link
        
    Returns:
        str: HTML download link
    """
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

def sentiment_visualization(textblob_sent: Dict, vader_sent: Dict) -> None:
    """
    Create visualizations for sentiment analysis results
    
    Args:
        textblob_sent (Dict): TextBlob sentiment results
        vader_sent (Dict): VADER sentiment results
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.bar(['Polarity', 'Subjectivity'], 
                [textblob_sent['polarity'], textblob_sent['subjectivity']], 
                color=['skyblue', 'salmon'])
        ax1.set_ylim(-1, 1)
        ax1.set_title('TextBlob Sentiment')
        
        vader_scores = {k: v for k, v in vader_sent.items() if k != 'compound'}
        ax2.bar(vader_scores.keys(), vader_scores.values())
        ax2.set_ylim(0, 1)
        ax2.set_title('VADER Sentiment')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating sentiment visualization: {e}")

def create_detailed_sentiment_analysis(text: str, sentiment_analyzer) -> None:
    """
    Create detailed sentiment analysis with progression chart
    
    Args:
        text (str): Input text
        sentiment_analyzer: VADER sentiment analyzer
    """
    try:
        sentences = sent_tokenize(text)
        if len(sentences) < 5:
            st.info("Not enough sentences for detailed sentiment analysis")
            return
        
        sentiments = [sentiment_analyzer.polarity_scores(s) for s in sentences]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot([s['compound'] for s in sentiments])
        ax.set_title('Sentiment Progression Throughout Video')
        ax.set_xlabel('Sentence Index')
        ax.set_ylabel('Sentiment Score')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.fill_between(range(len(sentiments)), [s['compound'] for s in sentiments], alpha=0.3)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        logger.error(f"Error creating detailed sentiment analysis: {e}")

def main():
    """Main Streamlit application"""
    st.title("🎬 Enhanced YouTube Video Summarizer")
    st.markdown("""
    This tool extracts and summarizes YouTube video transcripts, providing:
    - 📝 Concise summary
    - 🔑 Key keywords and topics
    - 😊 Sentiment analysis
    - 📥 Export options
    """)
    
    # Load models once
    try:
        models = load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Using fallback summarization methods")
        models = {
            'summarizer': None,
            'sentiment_analyzer': SentimentIntensityAnalyzer()
        }
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        max_summary_length = st.slider(
            "Max Summary Length:", 
            100, 2000, DEFAULT_SUMMARY_LENGTH
        )
        show_details = st.checkbox("Show detailed processing", False)
    
    video_url = st.text_input("Enter YouTube Video URL:", "")
    
    if st.button("Analyze Video"):
        if not video_url.strip():
            st.warning("Please enter a YouTube URL")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Extracting video ID...")
            video_id = extract_video_id(video_url)
            progress_bar.progress(10)
            
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return
            
            status_text.text("Fetching video metadata...")
            metadata = get_video_metadata(video_id)
            progress_bar.progress(20)
            
            if metadata:
                st.subheader(metadata['title'])
                st.caption(f"👤 {metadata['author']} | ⏱️ {metadata['length']} | 👀 {metadata['views']} views | 📅 {metadata.get('publish_date', 'Unknown')}")
            else:
                st.warning("Could not fetch video metadata")
            
            status_text.text("Fetching transcript...")
            transcript = get_cached_transcript(video_id)
            progress_bar.progress(40)
            
            if not transcript:
                st.error("Transcript not available for this video.")
                return
            
            # Check transcript quality
            quality_score = check_transcript_quality(transcript)
            if quality_score < 1.5:
                st.warning(f"Transcript quality is low (score: {quality_score:.2f}). Results may be less accurate.")
            
            status_text.text("Processing transcript...")
            video_text = ' '.join([line['text'] for line in transcript])
            progress_bar.progress(60)
            
            if not video_text.strip():
                st.error("Transcript appears to be empty.")
                return
            
            # Detect language
            lang = detect_language(video_text)
            if lang != 'en':
                st.info(f"Detected language: {lang}. Note: Analysis is optimized for English content.")
            
            if show_details:
                with st.expander("Raw Transcript"):
                    st.text(video_text[:2000] + ("..." if len(video_text) > 2000 else ""))
            
            status_text.text("Generating summary...")
            summary = summarize_text(video_text, models['summarizer'], max_summary_length)
            progress_bar.progress(70)
            
            status_text.text("Extracting keywords...")
            keywords = extract_keywords(video_text)
            progress_bar.progress(80)
            
            status_text.text("Analyzing topics...")
            topics = topic_modeling(video_text)
            progress_bar.progress(90)
            
            status_text.text("Performing sentiment analysis...")
            blob_sentiment = TextBlob(video_text).sentiment
            vader_sentiment = models['sentiment_analyzer'].polarity_scores(video_text)
            progress_bar.progress(100)
            
            st.subheader("📝 Summary")
            st.write(summary)
            
            st.subheader("🔑 Top Keywords")
            st.write(", ".join(keywords))
            
            st.subheader("🗂️ Detected Topics")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")
            
            st.subheader("😊 Sentiment Analysis")
            sentiment_visualization(
                {'polarity': blob_sentiment.polarity, 
                 'subjectivity': blob_sentiment.subjectivity},
                vader_sentiment
            )
            
            st.subheader("📥 Export Options")
            
            # Text export
            st.markdown(create_download_link(
                summary, 
                "summary.txt", 
                "Download Summary as TXT"
            ), unsafe_allow_html=True)
            
            # CSV export
            export_data = pd.DataFrame({
                "Summary": [summary],
                "Keywords": [', '.join(keywords)],
                "Topics": [' | '.join([', '.join(t) for t in topics])],
                "Polarity": [blob_sentiment.polarity],
                "Subjectivity": [blob_sentiment.subjectivity],
                "VADER_Positive": [vader_sentiment['pos']],
                "VADER_Negative": [vader_sentiment['neg']],
                "VADER_Neutral": [vader_sentiment['neu']],
                "VADER_Compound": [vader_sentiment['compound']],
                "Transcript_Quality": [quality_score],
                "Detected_Language": [lang]
            })
            csv = export_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="analysis.csv">Download Full Analysis as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            status_text.text("Analysis complete!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
