import os
from dotenv import load_dotenv
from sentiment_analysis import SentimentAnalysis
from translation_models import Translations
from ner_class import NER

load_dotenv()

# First exercise params
sentiment_model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_file = 'tiny_movie_reviews_dataset.txt'

# Second exercise params
data_folder = './data'
column_format = {0: 'text', 1: 'ner'}
train_file = 'train'
test_file = 'test'
dev_file = 'dev'
output_dir = 'train_output'
n_examples_to_train = 27
hidden_size = 256
epochs = 20

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
print("SECOND EXERCISE")
ner = NER(data_folder,column_format,train_file,test_file,dev_file,output_dir,n_examples_to_train)
ner.train(hidden_size,epochs)
ner.plot()
print('---------------------------------------------------------------')
print('\n')
print('---------------------------------------------------------------')
print("THIRD EXERCISE")
t = Translations(source_file, target_file, deepl_api_key)
t.deepl_translation()
t.google_translation()
print('---------------------------------------------------------------')
print('\n')