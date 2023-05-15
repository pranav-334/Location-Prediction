import os
import re
import tweepy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sqlite3

consumer_key = '733233798129823748-XmeWdyO43RrWe5gPlC1ZwDZ94GMPNAE'
consumer_secret = '7sSg4T8ijYpcZvpzcyCImvSAvJRSvvLYyNi3QxYs2knvq'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAFsbmAEAAAAAMNH%2Fpn3KyJczLZp%2BxgtjQ3kYIX8%3DQ172y7KsTFmI2zw9bvwuuFl7D4BEIwjnXGJQBFxQlfjgbKu9eU'

# Authenticate with the Twitter API
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
# auth = tweepy.AppAuthHandler(bearer_token=bearer_token)

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True)


## getting authentication error :
 # raise TweepyException('Expected token_type to equal "bearer", '
# tweepy.errors.TweepyException: Expected token_type to equal "bearer", but got None instead
#  So checked documentiation: https://coveralls.io/builds/38687818/source?filename=tweepy%2Fauth.py
# line 150:
# Cannot authenticate


# Create API object
# api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)



# consumer_key = 'aKBt8eJagd4PumKz8LGmZw'
# consumer_secret = 'asFAO5b3Amo8Turjl2RxiUVXyviK6PYe1X6sVVBA'
# access_token = '1914024835-dgZBlP6Tn2zHbmOVOPHIjSiTabp9bVAzRSsKaDX'
# access_token_secret = 'zCgN7F4csr6f3eU5uhX6NZR12O5o6mHWgBALY9U4'
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)
#
# # Create API object
# api = tweepy.API(auth, wait_on_rate_limit=True)
# uu = api.get_user(screen_name='DinakarTalari')


# Function to collect tweet data for a given location
def collect_tweets(location, count):
    tweets = []
    try:
        query = f"place:{location}"
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang='en').items(count):
            if 'retweeted_status' in tweet._json:
                tweets.append(clean_tweet(tweet.retweeted_status.full_text))
            else:
                tweets.append(clean_tweet(tweet.full_text))
    except TypeError as e:
        print("Error:", str(e))
    return tweets


def clean_tweet(self, text):
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Collect tweet data for different locations
location_tweets = {}
location_tweets['Hyderabad'] = collect_tweets('Hyderabad', 10)  # Replace with your desired location and hashtag
location_tweets['Banglore'] = collect_tweets('Banglore', 10)  # Replace with your desired location and hashtag

# Create a DataFrame from the collected tweets
tweet_data = pd.DataFrame({'tweet_text': [], 'location': []})
for location, tweets in location_tweets.items():
    tweet_data = tweet_data.append(pd.DataFrame({'tweet_text': tweets, 'location': location}))


# Split the dataset into training and testing sets
X = tweet_data['tweet_text']  # Tweet text as input
y = tweet_data['location']  # Location as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorized
vectorized = TfidfVectorizer()
X_train_vectorized = vectorized.fit_transform(X_train)
X_test_vectorized = vectorized.transform(X_test)


# Train and predict using Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
nb_predictions = nb_model.predict(X_test_vectorized)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)


# Train and predict using Support Vector Machine
svm_model = SVC()
svm_model.fit(X_train_vectorized, y_train)
svm_predictions = svm_model.predict(X_test_vectorized)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Support Vector Machine Accuracy:", svm_accuracy)

# Show the predicted locations for the test tweets
print("=== Predicted Locations ===")
for i, tweet in enumerate(X_test):
    print("Tweet:", tweet)
    print("Predicted Location:", svm_predictions[i])
    print("------------------------------")


# Train and predict using Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_vectorized, y_train)
dt_predictions = dt_model.predict(X_test_vectorized)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)


# Save results to a local SQLite database
connection = sqlite3.connect('predictions.db')
cursor = connection.cursor()

# Create a table for storing the predictions if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                  (tweet_text TEXT, predicted_location TEXT)''')

# Insert the predicted locations into the table
for i, tweet in enumerate(X_test):
    query = "INSERT INTO predictions (tweet_text, predicted_location) VALUES (?, ?)"
    values = (tweet, svm_predictions[i])
    cursor.execute(query, values)

# Commit the changes and close the database connection
connection.commit()
connection.close()

print("Predictions saved to the database.")
