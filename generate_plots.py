import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def prepare_data():
    """
    Load location and review data and extract reviews of top 5 and bottom 5
    rated locations
    """
    # Load data:
    locations = pd.read_csv("location_data.csv")
    reviews = pd.read_csv("technicalAssessment-GoogleReviews.csv")

    # Merge datasets on storeID:
    df = pd.merge(reviews, locations, on="storeID")

    # Get the restaurants with the highest and lowest ratings:
    restaurant_stats = df.groupby("locationName")\
        .agg({"overallRating": "mean", "reviewDate": "count"})\
        .sort_values(by="overallRating", ascending=False)
    # Filter to restaurants with more than 10 reviews:
    restaurant_stats = restaurant_stats[restaurant_stats["reviewDate"] > 10]
    # Get restaurant names:
    top_5 = restaurant_stats.head().index.tolist()
    bottom_5 = restaurant_stats.tail().index.tolist()
    # Get reviews:
    top_5_reviews = df[df["locationName"].isin(top_5)]["reviewText"].dropna()
    bottom_5_reviews = df[df["locationName"].isin(bottom_5)]["reviewText"]\
        .dropna()

    return (top_5_reviews, bottom_5_reviews)


def run_topic_model(text_data, stop_words):
    """
    Create a pipeline for topic modeling
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_df=0.95,
                                  min_df=2,
                                  stop_words=stop_words)),
        ("lda", LatentDirichletAllocation(n_components=3, random_state=42)),
    ])
    pipeline.fit(text_data)
    return pipeline


# Define a function to plot topics for a specific location group
def plot_topics(model, feature_names, n_top_words, file_name, bad_words):
    """
    Given a topic model, plot the 2nd and 3rd most prevalent topics
    """
    n_topics = model.components_[1:].shape[0]

    fig, axes = plt.subplots(n_topics,
                             1,
                             figsize=(10, 3 * n_topics),
                             sharex=True)

    for topic_idx, ax in enumerate(axes):
        topic = model.components_[1:][topic_idx]
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        colors = ["#0674cc" if word not in bad_words else "red" \
                  for word in top_words]
        ax.barh(top_words, topic[top_words_idx], color=colors)
        ax.set_title(f"Topic {topic_idx+1}")
        ax.invert_yaxis()

    plt.xlabel("Weight")
    plt.tight_layout()
    plt.savefig("plots/" + file_name)
    return


def main():
    # Get reviews:
    top_5_reviews, bottom_5_reviews = prepare_data()

    # Add custom stop words:
    added_stop_words = ["food", "service", "atmosphere", "yes", "la",
                        "madeleine", "5recommended"]
    stop_words = stopwords.words("english") + added_stop_words

    # Run topic model on top 5 and bottom 5 restaurant reviews:
    top_5_model = run_topic_model(top_5_reviews, stop_words)
    bottom_5_model = run_topic_model(bottom_5_reviews, stop_words)

    # Get feature names:
    tfidf_feature_names_top = top_5_model.named_steps["tfidf"]\
        .get_feature_names_out()
    tfidf_feature_names_bottom = bottom_5_model.named_steps["tfidf"]\
        .get_feature_names_out()

    bad_words = set(["rude", "bad"])

    # Plot topics for top 5 locations
    plot_topics(top_5_model.named_steps["lda"],
                tfidf_feature_names_top,
                n_top_words=9,
                file_name="top_5_LDA.pdf",
                bad_words=bad_words)

    # Plot topics for bottom 5 locations
    plot_topics(bottom_5_model.named_steps["lda"],
                tfidf_feature_names_bottom,
                n_top_words=9,
                file_name = "bottom_5_LDA.pdf",
                bad_words=bad_words)
    return


if __name__ == "__main__":
    main()
