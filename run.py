import os
from dotenv import load_dotenv
from sentiment_analysis import SentimentAnalysis
from translation_models import Translations

load_dotenv()

# First exercise params
sentiment_model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_file = 'tiny_movie_reviews_dataset.txt'

# Third exercise params
source_file = 'spanish.txt'
target_file = 'english.txt'
deepl_api_key = os.environ['DEEPL_API_KEY']

print('---------------------------------------------------------------')
print("FIRST EXERCISE")
s = SentimentAnalysis(sentiment_model, sentiment_file)
s.run()
print('---------------------------------------------------------------')
print('\n')
print('---------------------------------------------------------------')
print("THIRD EXERCISE")
t = Translations(source_file, target_file, deepl_api_key)
t.deepl_translation()
t.google_translation()
print('---------------------------------------------------------------')
print('\n')