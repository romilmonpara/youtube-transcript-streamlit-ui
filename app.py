# Title : Task-A - Youtube Transcript Summrizer with Sentiment Analysis
# Name : Monpara Romil Kamleshbhai
# Enroll No.: 23002170210064

from typing import Counter
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
import re
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # Need for lemmatization

def summarize_text(text, max_length=1024):
    summarization_pipeline = pipeline("summarization")

    # Ensure text is within model limits
    if len(text) < 50:
        return "Text is too short to summarize."

    text = text[:4000]  # Adjust according to the model's limits

    try:
        summary = summarization_pipeline(text, max_length=min(len(text)//2, 512), min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error in summarization: {str(e)}"


def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    words = re.findall(r'\b\w+\b', text.lower())  
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 1]

    word_freq = Counter(words)
    
    if not word_freq:
        return ["No significant keywords found"]

    top_keywords = [word for word, _ in word_freq.most_common(5)]  
    return top_keywords

def topic_modeling(text):
    if len(text.split()) < 5:  # Ensure enough words for topic modeling
        return ["Not enough text for topic modeling"]

    try:
        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        tf = vectorizer.fit_transform([text])

        if tf.shape[1] == 0:  # Check if vocabulary is empty
            return ["Not enough unique words for topic modeling"]

        lda_model = LatentDirichletAllocation(n_components=3, max_iter=5, learning_method='online', random_state=42)
        lda_model.fit(tf)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])

        return topics
    except Exception as e:
        return [f"Topic modeling failed: {str(e)}"]

def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',  # Pattern for URLs with 'v=' parameter
        r'youtu.be/([^?]+)',  # Pattern for shortened URLs
        r'youtube.com/embed/([^?]+)'  # Pattern for embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    return video_id

# Streamlit App
def main():
    st.title("YouTube Video Summarizer (English)")
    st.header("Welcome to the YouTube Transcript Summarizer!")
    st.write("This tool extracts and summarizes YouTube video transcripts, providing keywords, topics, and sentiment analysis.")

    video_url = st.text_input("Enter YouTube Video URL:", "")

    max_summary_length = st.slider("Max Summary Length:", 100, 2000, 500)

    if st.button("Summarize"):
        try:
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please enter a valid URL.")
                return

            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            if not transcript:
                st.error("Transcript not available for this video.")
                return

            video_text = ' '.join([line['text'] for line in transcript])

            if not video_text.strip():  # Check if transcript is empty
                st.error("Transcript appears to be empty.")
                return

            summary = summarize_text(video_text, max_length=max_summary_length)
            keywords = extract_keywords(video_text)
            topics = topic_modeling(video_text)
            sentiment = TextBlob(video_text).sentiment

            st.subheader("Video Summary:")
            st.write(summary)

            st.subheader("Keywords:")
            st.write(keywords)

            st.subheader("Topics:")
            for idx, topic in enumerate(topics):
                st.write(f"Topic {idx+1}: {', '.join(topic)}")

            st.subheader("Sentiment Analysis:")
            st.write(f"Polarity: {sentiment.polarity}")
            st.write(f"Subjectivity: {sentiment.subjectivity}")

        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except NoTranscriptFound:
            st.error("No transcript found for this video.")
        except Exception as e:
            import traceback
            st.error(f"Error: {str(e)}")
            st.text(traceback.format_exc())  # Print full traceback for debugging

if __name__ == "__main__":
    main()
