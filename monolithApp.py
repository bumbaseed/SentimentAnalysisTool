from flask import Flask, request, render_template, url_for, session, redirect
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from wordcloud import WordCloud
import logging
from rake_nltk import Rake
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # secret key for session management

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define available models
models = {
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "bert-base": "nlptown/bert-base-multilingual-uncased-sentiment"
}

def analyze_text(text, pipe):
    """Function to perform sentiment analysis on a single text"""
    result = pipe(text, truncation=True, max_length=512)
    return {'text': text, 'sentiment': result}

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def generate_wordcloud(texts, output_path):
    if not texts:
        logging.warning("No texts available for word cloud generation.")
        return
    text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, format='png')
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    results = []
    texts = []
    insights = []

    # Get form data
    text = request.form.get('text', '')
    file = request.files.get('file')
    file_type = request.form.get('file_type', '')
    model_name = request.form.get('model_select', '')  

    logging.info(f"Model selected from form: {model_name}")

    # Validate and select the model
    if model_name in models:
        model_path = models[model_name]
        pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)
        logging.info(f"Using model: {model_name} - {model_path}")
    else:
        logging.error("Invalid model selection")
        return "Invalid model selection", 400
    
    all_entities = []

    # Process text input
    if text:
        result = analyze_text(text, pipe)
        keywords = extract_keywords(text)
        entities = perform_ner(text)
        insights.append({'text': text, 'keywords': keywords, 'entities': entities})
        results.append(result)
        texts.append(text)
        all_entities.extend(entities)

    # Process file upload
    if file:
        logging.info(f"Processing file of type: {file_type}")
        if file_type == 'txt':
            file_content = file.read().decode('utf-8')
            for line in file_content.splitlines():
                if line.strip():  # Ensure the line is not empty
                    result = analyze_text(line, pipe)
                    keywords = extract_keywords(line)
                    entities = perform_ner(line)
                    insights.append({'text': line, 'keywords': keywords, 'entities': entities})
                    results.append(result)
                    texts.append(line)
                    all_entities.extend(entities)
        elif file_type == 'csv':
            df = pd.read_csv(file)
            column = request.form.get('text_column', 'text')
            logging.info(f"Reading CSV file, looking for column: {column}")
            if column in df.columns:
                for text in df[column].dropna().values:
                    if text.strip():  # Ensure the text is not empty
                        result = analyze_text(text, pipe)
                        keywords = extract_keywords(text)
                        entities = perform_ner(text)
                        insights.append({'text': text, 'keywords': keywords, 'entities': entities})
                        results.append(result)
                        texts.append(text)
                        all_entities.extend(entities)
            else:
                logging.error(f"Column '{column}' not found in CSV.")
                return f"Column '{column}' not found in CSV.", 400
        elif file_type == 'json':
            data = json.load(file)
            logging.info("Reading JSON file")
            for entry in data:
                for key, value in entry.items():
                    if isinstance(value, str) and value.strip():  # Ensure the value is a non-empty string
                        result = analyze_text(value, pipe)
                        keywords = extract_keywords(value)
                        entities = perform_ner(value)
                        insights.append({'text': value, 'keywords': keywords, 'entities': entities})
                        results.append(result)
                        texts.append(value)
                        all_entities.extend(entities)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return f"Unsupported file type: {file_type}", 400

    if not text and not file:
        logging.error("No text input or file uploaded")
        return "No text input or file uploaded", 400

    # Check if texts list is empty before generating word cloud
    if not texts:
        logging.error("No valid texts were found for analysis")
        return "No valid texts were found for analysis", 400

    wordcloud_path = os.path.join('static', 'wordcloud.png')
    generate_wordcloud(texts, wordcloud_path)

    sentiment_counts = {}
    sentiment_scores = {}
    for result in results:
        for sentiment in result['sentiment']:
            label = sentiment['label']
            score = sentiment['score']
            if label not in sentiment_counts:
                sentiment_counts[label] = 0
                sentiment_scores[label] = []
            sentiment_counts[label] += 1
            sentiment_scores[label].append(score)

    logging.info(f"Sentiment counts: {sentiment_counts}")

    # Calculate overall sentiment for percentages
    total_sentiments = sum(sentiment_counts.values())
    logging.info(f"Total sentiments: {total_sentiments}")
    sentiment_percentages = {}
    if total_sentiments > 0:
        sentiment_percentages = {label: round((count / total_sentiments) * 100, 1) for label, count in sentiment_counts.items()}

    logging.info(f"Sentiment percentages: {sentiment_percentages}")

    # Get top positive and negative texts
    top_positive_texts = []
    top_negative_texts = []
    for item in results:
        for sentiment in item['sentiment']:
            label = sentiment['label']
            score = sentiment['score']
            if label == 'LABEL_2':  
                top_positive_texts.append((item['text'], score))
            elif label == 'LABEL_0':
                top_negative_texts.append((item['text'], score))
                
    # Sort the texts by score
    top_positive_texts.sort(key=lambda x: x[1], reverse=True)
    top_negative_texts.sort(key=lambda x: x[1], reverse=True)
    top_positive_texts = top_positive_texts[:5]
    top_negative_texts = top_negative_texts[:5]

    # Bar chart of sentiment counts
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['red', 'blue', 'green'])
    chart_path_1 = os.path.join('static', 'sentiment_bar_chart.png')
    plt.savefig(chart_path_1)
    plt.close()

    # Histogram of sentiment scores
    plt.figure(figsize=(10, 6))
    for label, scores in sentiment_scores.items():
        plt.hist(scores, bins=50, alpha=0.5, label=label)
    plt.legend(loc='upper right')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Sentiment Score Distribution')
    chart_path_2 = os.path.join('static', 'sentiment_score_distribution.png')
    plt.savefig(chart_path_2)
    plt.close()

    session['insights'] = insights
    session['sentiment_percentages'] = sentiment_percentages
    session['all_entities'] = all_entities

    return render_template('results.html', results=results, chart_url_1=url_for('static', filename='sentiment_bar_chart.png'), chart_url_2=url_for('static', filename='sentiment_score_distribution.png'), top_positive_texts=top_positive_texts, top_negative_texts=top_negative_texts, wordcloud_url=url_for('static', filename='wordcloud.png'), model_name=model_name)

@app.route('/insights')
def insights():
    insights = session.get('insights', [])
    sentiment_percentages = session.get('sentiment_percentages', {})
    
    all_entities = session.get('all_entities', [])
    return render_template('insights.html', insights=insights, sentiment_percentages=sentiment_percentages, all_entities=all_entities)

@app.route('/charts')
def charts():
    return render_template('charts.html', chart_url_1=url_for('static', filename='sentiment_bar_chart.png'), chart_url_2=url_for('static', filename='sentiment_score_distribution.png'), wordcloud_url=url_for('static', filename='wordcloud.png'))

if __name__ == "__main__":
    app.run(debug=True)
