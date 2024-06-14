from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import spacy
from rake_nltk import Rake
import logging

nlp = spacy.load("en_core_web_sm")

def analyze_text(text, pipe):
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

def get_sentiment_stats(results):
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
    return sentiment_counts, sentiment_scores

def calculate_percentages(sentiment_counts):
    total_sentiments = sum(sentiment_counts.values())
    sentiment_percentages = {}
    if total_sentiments > 0:
        sentiment_percentages = {label: round((count / total_sentiments) * 100, 1) for label, count in sentiment_counts.items()}
    return sentiment_percentages

def get_top_texts(results):
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
    top_positive_texts.sort(key=lambda x: x[1], reverse=True)
    top_negative_texts.sort(key=lambda x: x[1], reverse=True)
    return top_positive_texts[:5], top_negative_texts[:5]

def generate_charts(sentiment_counts, sentiment_scores):
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=['red', 'blue', 'green'])
    chart_path_1 = os.path.join('static', 'sentiment_bar_chart.png')
    plt.savefig(chart_path_1)
    plt.close()

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
    
    return chart_path_1, chart_path_2

def select_model(model_name):
    models = {
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "bert-base": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    if model_name in models:
        model_path = models[model_name]
        pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)
        return model_path, pipe
    return None, None

def process_text(text, pipe):
    result = analyze_text(text, pipe)
    keywords = extract_keywords(text)
    entities = perform_ner(text)
    return result, keywords, entities
