from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging



# Dont need to do this for the homework, but just FYI and as a way to improve your code in general: 
# https://docs.python.org/3/library/typing.html
# Adding typing can really help with catching small errors! 
# 
# Also, using black for autoformatting: 
# https://pypi.org/project/black/


class SentimentAnalysis:
    '''
    Class that solves the first problem
    '''

    def __init__(self, model, file):
        '''
        Constructor of the class, receives the model to be used and the data to be analyzed
        '''
        logging.set_verbosity_error()
        # should be single underscores for the next 4 lines: https://towardsdatascience.com/whats-the-meaning-of-single-and-double-underscores-in-python-3d27d57d6bd1
        self.__model = AutoModelForSequenceClassification.from_pretrained(model)
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__file = file
        self.__classifications = {'Negative': 'NEGATIVE', 'Positive': 'POSITIVE', 'Neutral': 'POSITIVE'}


    def run(self):
        '''
        Runs the sentiment analysis on the dataset
        '''
        with open(self.__file, 'r') as f:
            for line in f:
                sentiment_task = pipeline("sentiment-analysis", model=self.__model, tokenizer=self.__tokenizer)
                print(self.__classifications[sentiment_task(line)[0]['label']])


# TESTS
if __name__ == '__main__':
    
"""Be sure to add tests for all pieces of functionality! 
Tests should check that the functionality/outputs are as expected, NOT just that the code runs! 
When writing code, I write the test for each class or piece of functionality as soon as I finish that piece.
It helps you develop incrementally, being sure that each piece of code is clean and works like you expect it to! 
for tests, best practices are to have a structure like this: 
https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
"""
    s = SentimentAnalysis('cardiffnlp/twitter-roberta-base-sentiment-latest', 'tiny_movie_reviews_dataset.txt')
    s.run()
