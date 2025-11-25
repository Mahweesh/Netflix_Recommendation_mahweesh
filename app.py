from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load CSV file
train_data = pd.read_csv("models/netflix_titles_updated.csv")

# Fill NaN values for relevant columns
for col in ['title', 'description', 'director']:
    train_data[col] = train_data[col].fillna('')

# Create 'Tags' column for TF-IDF
train_data['Tags'] = (train_data['title'] + ' ' +
                      train_data['description'] + ' ' +
                      train_data['director']).str.lower()


# Helper function to truncate long text
def truncate(text, length=100):
    return text[:length] + "..." if len(text) > length else text


# Keyword-based recommendations
def keyword_based_recommendations(train_data, keyword, top_n=10):
    keyword = str(keyword).lower().strip()

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(train_data['Tags'])

    query_vec = tfidf.transform([keyword])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = cosine_sim.argsort()[::-1][:top_n]
    recommended = train_data.iloc[top_indices].copy()
    recommended['similarity'] = cosine_sim[top_indices]
    recommended = recommended[recommended['similarity'] > 0]

    return recommended[['title', 'rating_num', 'description', 'director']]


# Content-based recommendations
def content_based_recommendations(train_data, item_name, top_n=10):
    item_name = str(item_name).lower()
    matches = train_data[train_data['title'].str.lower() == item_name]

    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(train_data['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sim_scores[1:top_n + 1]]

    return train_data.iloc[top_indices][['title', 'rating_num', 'description', 'director']]


# Routes
@app.route("/")
def index():
    return render_template("main.html",
                           content_based_rec=pd.DataFrame(),
                           truncate=truncate,
                           message=None)


@app.route("/recommendations", methods=["POST"])
def recommendations():
    keyword = request.form.get("title")
    nbr = int(request.form.get("nbr"))

    recs = keyword_based_recommendations(train_data, keyword, top_n=nbr)

    message = None
    if recs.empty:
        message = f"No recommendations found for keyword: {keyword}"

    return render_template("main.html",
                           content_based_rec=recs,
                           truncate=truncate,
                           message=message)


if __name__ == "__main__":
    app.run(debug=True)
