from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging


class SentimentAnalysis:
    '''
    Class that solves the first problem
    '''

    def __init__(self, model, file):
        '''
        Constructor of the class, receives the model to be used and the data to be analyzed
        '''
        logging.set_verbosity_error()
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._file = file
        self._classifications = {'negative': 'NEGATIVE', 'positive': 'POSITIVE', 'neutral': 'POSITIVE'}


    def run(self):
        '''
        Runs the sentiment analysis on the dataset
        '''
        with open(self._file, 'r') as f:
            for line in f:
                sentiment_task = pipeline("sentiment-analysis", model=self._model, tokenizer=self._tokenizer)
                print(self._classifications[sentiment_task(line)[0]['label']])


# TESTS
if __name__ == '__main__':
    s = SentimentAnalysis('cardiffnlp/twitter-roberta-base-sentiment-latest', 'tiny_movie_reviews_dataset.txt')
    s.run()