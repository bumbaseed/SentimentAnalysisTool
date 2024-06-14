import pandas as pd
import json
from analysis.sentiment_analysis import process_text


def process_file(file, file_type, pipe, text_column='text'):
    results = []
    insights = []
    texts = []
    all_entities = []
    
    if file_type == 'txt':
        file_content = file.read().decode('utf-8')
        for line in file_content.splitlines():
            if line.strip():
                result, keywords, entities = process_text(line, pipe)
                insights.append({'text': line, 'keywords': keywords, 'entities': entities})
                results.append(result)
                texts.append(line)
                all_entities.extend(entities)
    elif file_type == 'csv':
        df = pd.read_csv(file)
        if text_column in df.columns:
            for text in df[text_column].dropna().values:
                if text.strip():
                    result, keywords, entities = process_text(text, pipe)
                    insights.append({'text': text, 'keywords': keywords, 'entities': entities})
                    results.append(result)
                    texts.append(text)
                    all_entities.extend(entities)
        else:
            return f"Column '{text_column}' not found in CSV."
    elif file_type == 'json':
        data = json.load(file)
        for entry in data:
            for key, value in entry.items():
                if isinstance(value, str) and value.strip():
                    result, keywords, entities = process_text(value, pipe)
                    insights.append({'text': value, 'keywords': keywords, 'entities': entities})
                    results.append(result)
                    texts.append(value)
                    all_entities.extend(entities)
    else:
        return f"Unsupported file type: {file_type}"
    
    return {'results': results, 'insights': insights, 'texts': texts, 'all_entities': all_entities}
