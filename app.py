from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

pos_reviews = []
neg_reviews = []
neu_reviews = []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET', 'POST'])
def process():
    text = request.form['text']
    score = sia.polarity_scores(text).get('compound')
    if score > 0:
        sentiment = 'positive'
    elif score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return render_template('result.html', text=text, score=score, sentiment=sentiment)

@app.route('/show_reviews/<review_type>', methods=['GET', 'POST'])
def show_reviews(review_type):
    if review_type == 'positive':
        reviews = pos_reviews
        title = 'Positive Reviews'
        color = '#2ecc71'
    elif review_type == 'negative':
        reviews = neg_reviews
        title = 'Negative Reviews'
        color = '#e74c3c'
    elif review_type == 'neutral':
        reviews = neu_reviews
        title = 'Neutral Reviews'
        color = '#3498db'
    else:
        return "Invalid review type"
    return render_template('show_reviews.html', reviews=reviews, title=title, color=color)

@app.route('/process2', methods=['POST', 'GET'])
def process2():
    if request.method == 'POST':
        file = request.files.get('file', False)
        if file:
            pos_reviews.clear()
            neg_reviews.clear()
            neu_reviews.clear()

            df = pd.read_csv(file)
            num_pos = 0
            num_neg = 0
            num_neu = 0
            pos_score = []
            neg_score = []
            neu_score = []

            for text in df['ReviewContent']:
                score = sia.polarity_scores(str(text)).get('compound')
                if score > 0:
                    num_pos += 1
                    pos_score.append(score)
                    pos_reviews.append(text)
                elif score < 0:
                    num_neg += 1
                    neg_score.append(score)
                    neg_reviews.append(text)
                else:
                    num_neu += 1
                    neu_score.append(score)
                    neu_reviews.append(text)

            if num_pos >= num_neg and num_pos >= num_neu:
                final_sentiment = 'positive'
                final_sentiment_score = sum(pos_score) / len(pos_score)
            elif num_neg >= num_pos and num_neg >= num_neu:
                final_sentiment = 'negative'
                final_sentiment_score = sum(neg_score) / len(neg_score)
            else:
                final_sentiment = 'neutral'
                final_sentiment_score = sum(neu_score) / len(neu_score)

            return render_template('result2.html', num_pos=num_pos,
                                   num_neg=num_neg, num_neu=num_neu,
                                   final_sentiment=final_sentiment,
                                   final_sentiment_score=final_sentiment_score)
        else:
            return "No file uploaded."
    else:
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

