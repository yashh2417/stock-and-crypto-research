from flask import Flask, render_template, request, send_file
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from bs4 import BeautifulSoup
import requests, re, torch, pandas as pd
import os

app = Flask(__name__)

# Load summarization model
pegasus_tokenizer = PegasusTokenizer.from_pretrained("human-centered-summarization/financial-summarization-pegasus")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("human-centered-summarization/financial-summarization-pegasus")

# Load sentiment models
bert_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
bert_model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")

albert_tokenizer = AlbertTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
albert_model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")

exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support', 'search']

# Helpers
def search_news_urls(ticker):
    url = f"https://www.google.com/search?q=google+finance+{ticker}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    return [a['href'] for a in soup.find_all('a', href=True)]

def clean_urls(urls):
    cleaned = []
    for url in urls:
        if 'https://' in url and not any(e in url for e in exclude_list):
            found = re.findall(r'(https?://\S+)', url)
            if found:
                cleaned.append(found[0].split('&')[0])
    return list(set(cleaned))

def scrape_articles(urls):
    articles = []
    for url in urls[:5]:
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            content = ' '.join(paragraphs).split(' ')[:350]
            articles.append(' '.join(content))
        except:
            articles.append("Could not retrieve content.")
    return articles

def summarize_articles(articles):
    summaries = []
    for article in articles:
        input_ids = pegasus_tokenizer.encode(article, return_tensors="pt", truncation=True)
        output = pegasus_model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = pegasus_tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def analyze_sentiment(texts, tokenizer, model):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        label = model.config.id2label[pred] if hasattr(model.config, 'id2label') else str(pred)
        score = torch.softmax(logits, dim=1)[0][pred].item()
        results.append({"label": label, "score": score})
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        tickers_input = request.form.get("tickers", "")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        data = []

        for ticker in tickers:
            raw_urls = search_news_urls(ticker)
            cleaned = clean_urls(raw_urls)
            articles = scrape_articles(cleaned)
            summaries = summarize_articles(articles)

            sentiment_bert = analyze_sentiment(summaries, bert_tokenizer, bert_model)
            sentiment_roberta = analyze_sentiment(summaries, roberta_tokenizer, roberta_model)
            sentiment_albert = analyze_sentiment(summaries, albert_tokenizer, albert_model)

            for i in range(len(summaries)):
                data.append({
                    "Ticker": ticker,
                    "Summary": summaries[i],
                    "BERT Sentiment": sentiment_bert[i]["label"],
                    "BERT Score": sentiment_bert[i]["score"],
                    "RoBERTa Sentiment": sentiment_roberta[i]["label"],
                    "RoBERTa Score": sentiment_roberta[i]["score"],
                    "ALBERT Sentiment": sentiment_albert[i]["label"],
                    "ALBERT Score": sentiment_albert[i]["score"],
                    "URL": cleaned[i] if i < len(cleaned) else "N/A"
                })

        df = pd.DataFrame(data)
        file_path = "sentiment_results.csv"
        df.to_csv(file_path, index=False)
        return send_file(file_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
