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
        self.__model = AutoModelForSequenceClassification.from_pretrained(model)
        self.__tokenizer = AutoTokenizer.from_pretrained(model)
        self.__file = file
        self.__classifications = {'Negative': 'NEGATIVE', 'Positive': 'POSITIVE', 'Neutral': 'POSITIVE'}

    def run(self):
        with open(self.__file, 'r') as f:
            for line in f:
                sentiment_task = pipeline("sentiment-analysis", model=self.__model, tokenizer=self.__tokenizer)
                print(self.__classifications[sentiment_task(line)[0]['label']])
