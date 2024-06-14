from flask import Blueprint, render_template, request, session, redirect, url_for
from analysis.sentiment_analysis import analyze_text, process_text, extract_keywords, perform_ner, generate_wordcloud, select_model, get_sentiment_stats, calculate_percentages, get_top_texts, generate_charts
from utils.file_processing import process_file
import logging

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    results = []
    texts = []
    insights = []

    text = request.form.get('text', '')
    file = request.files.get('file')
    file_type = request.form.get('file_type', '')
    model_name = request.form.get('model_select', '')

    logging.info(f"Model selected from form: {model_name}")

    # Validate and select the model
    model_path, pipe = select_model(model_name)
    if not pipe:
        logging.error("Invalid model selection")
        return "Invalid model selection", 400

    all_entities = []

    if text:
        result, keywords, entities = process_text(text, pipe)
        insights.append({'text': text, 'keywords': keywords, 'entities': entities})
        results.append(result)
        texts.append(text)
        all_entities.extend(entities)

    if file:
        file_results = process_file(file, file_type, pipe, request.form.get('text_column', 'text'))
        if isinstance(file_results, str):
            logging.error(file_results)
            return file_results, 400
        results.extend(file_results['results'])
        insights.extend(file_results['insights'])
        texts.extend(file_results['texts'])
        all_entities.extend(file_results['all_entities'])

    if not texts:
        logging.error("No valid texts were found for analysis")
        return "No valid texts were found for analysis", 400

    wordcloud_path = 'static/wordcloud.png'
    generate_wordcloud(texts, wordcloud_path)

    sentiment_counts, sentiment_scores = get_sentiment_stats(results)
    sentiment_percentages = calculate_percentages(sentiment_counts)

    top_positive_texts, top_negative_texts = get_top_texts(results)

    chart_path_1, chart_path_2 = generate_charts(sentiment_counts, sentiment_scores)

    session['insights'] = insights
    session['sentiment_percentages'] = sentiment_percentages
    session['all_entities'] = all_entities
    session['chart_url_1'] = url_for('static', filename='sentiment_bar_chart.png')
    session['chart_url_2'] = url_for('static', filename='sentiment_score_distribution.png')
    session['wordcloud_url'] = url_for('static', filename='wordcloud.png')

    return render_template('results.html', results=results, chart_url_1=url_for('static', filename='sentiment_bar_chart.png'), chart_url_2=url_for('static', filename='sentiment_score_distribution.png'), top_positive_texts=top_positive_texts, top_negative_texts=top_negative_texts, wordcloud_url=url_for('static', filename='wordcloud.png'), model_name=model_name)

@main.route('/insights')
def insights():
    insights = session.get('insights', [])
    sentiment_percentages = session.get('sentiment_percentages', {})
    logging.info(f"Retrieved sentiment percentages from session: {sentiment_percentages}")
    all_entities = session.get('all_entities', [])
    return render_template('insights.html', insights=insights, sentiment_percentages=sentiment_percentages, all_entities=all_entities)

@main.route('/charts')
def charts():
    chart_url_1 = session.get('chart_url_1')
    chart_url_2 = session.get('chart_url_2')
    wordcloud_url = session.get('wordcloud_url')
    return render_template('charts.html', chart_url_1=chart_url_1, chart_url_2=chart_url_2, wordcloud_url=wordcloud_url)

