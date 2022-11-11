from sentiment_analysis import SentimentAnalysis

# First model params
sentiment_model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_file = 'tiny_movie_reviews_dataset.txt'

print('---------------------------------------------------------------')
print("FIRST EXERCISE")
s = SentimentAnalysis(sentiment_model, sentiment_file)
s.run()
print('---------------------------------------------------------------')
print('\n')