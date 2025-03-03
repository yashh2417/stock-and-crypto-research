# 1. Install and Import Baseline Dependencies
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
import csv
import torch
import numpy as np
from gensim.models import KeyedVectors

# 2. Setup Models

# Summarization Models
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer_pegasus = PegasusTokenizer.from_pretrained(model_name)
model_pegasus = PegasusForConditionalGeneration.from_pretrained(model_name)

# Sentiment Analysis Models
model_name_bert = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer_bert = BertTokenizer.from_pretrained(model_name_bert)
model_bert = BertForSequenceClassification.from_pretrained(model_name_bert)

model_name_roberta = "roberta-large-mnli"
tokenizer_roberta = RobertaTokenizer.from_pretrained(model_name_roberta)
model_roberta = RobertaForSequenceClassification.from_pretrained(model_name_roberta)

model_name_albert = "textattack/albert-base-v2-SST-2"
tokenizer_albert = AlbertTokenizer.from_pretrained(model_name_albert)
model_albert = AlbertForSequenceClassification.from_pretrained(model_name_albert)

# 3. Building a News and Sentiment Pipeline
monitored_tickers = ['GME', 'TSLA', 'BTC']

# 4.1. Search for Stock News using Google and Yahoo Finance
print('Searching for stock news for', monitored_tickers)
# def search_for_stock_news_links(ticker):
#     search_url = 'https://www.google.com/search?q=yahoo+finance+{}&tbm=nws'.format(ticker)
#     r = requests.get(search_url)
#     soup = BeautifulSoup(r.text, 'html.parser')
#     atags = soup.find_all('a')
#     hrefs = [link['href'] for link in atags]
#     return hrefs

# Google finance
def search_for_stock_news_urls(ticker):
    search_url = "https://www.google.com/search?q=google+finance+{}&tbm=nws".format(ticker)
    r = requests.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]
    return hrefs

raw_urls = {ticker: search_for_stock_news_links(ticker) for ticker in monitored_tickers}

# 4.2. Strip out unwanted URLs
print('Cleaning URLs.')
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']
def strip_unwanted_urls(urls, exclude_list):
    val = []
    for url in urls:
        if 'https://' in url and not any(exc in url for exc in exclude_list):
            res = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            val.append(res)
    return list(set(val))

cleaned_urls = {ticker: strip_unwanted_urls(raw_urls[ticker], exclude_list) for ticker in monitored_tickers}

# 4.3. Search and Scrape Cleaned URLs
print('Scraping news links.')
def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        results = soup.find_all('p')
        text = [res.text for res in results]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker: scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

# 4.4. Summarize all Articles
print('Summarizing articles.')
def summarize(articles, model_name="pegasus"):
    summaries = []
    tokenizer, model = models[model_name]
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors="pt")
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {ticker: summarize(articles[ticker], model_name="pegasus") for ticker in monitored_tickers}

# 5. Adding Sentiment Analysis
print('Calculating sentiment.')
def analyze_sentiment(texts, tokenizer, model):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment = model.config.id2label[predicted_class]
        score = torch.softmax(logits, dim=1)[0][predicted_class].item()
        results.append({"label": sentiment, "score": score})
    return results

# Analyze sentiment for each ticker
sentiment_results = {
    "bert": {ticker: analyze_sentiment(summaries[ticker], tokenizer_bert, model_bert) for ticker in monitored_tickers},
    "roberta": {ticker: analyze_sentiment(summaries[ticker], tokenizer_roberta, model_roberta) for ticker in monitored_tickers},
    "albert": {ticker: analyze_sentiment(summaries[ticker], tokenizer_albert, model_albert) for ticker in monitored_tickers},
}

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame for visualization
data = []
for ticker in monitored_tickers:
    for i, summary in enumerate(summaries[ticker]):
        data.append({
            "Ticker": ticker,
            "Summary": summary,
            "BERT Sentiment": sentiment_results["bert"][ticker][i]["label"],
            "BERT Score": sentiment_results["bert"][ticker][i]["score"],
            "RoBERTa Sentiment": sentiment_results["roberta"][ticker][i]["label"],
            "RoBERTa Score": sentiment_results["roberta"][ticker][i]["score"],
            "ALBERT Sentiment": sentiment_results["albert"][ticker][i]["label"],
            "ALBERT Score": sentiment_results["albert"][ticker][i]["score"],
            "URL": cleaned_urls[ticker][i]
        })

df = pd.DataFrame(data)

print('Exporting results.')
df.to_csv("sentiment_analysis_google01.csv", index=False)
