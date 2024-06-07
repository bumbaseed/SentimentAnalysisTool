from flask import Flask, request, render_template
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification

app = Flask(__name__)

# Initialize the pipeline - THIS SETUP OUTPUTS CORRECT SCORES.
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    print(f"Analyzing text: {text}")
    
    # Perform sentiment analysis
    result = pipe(text, truncation=True, max_length=512)
    print(f"Result: {result}")
    
    return render_template('results.html', text=text, sentiment=result)

if __name__ == "__main__":
    app.run(debug=True)




    
