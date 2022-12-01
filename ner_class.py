from flair.data import Corpus, Dictionary
from flair.datasets.biomedical import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.visual.training_curves import Plotter
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import os


class NER:
    '''
    Class that solves the second problem
    '''

    def __init__(self, data_folder: str, column_format: dict, train_file: str, test_file: str, dev_file: str, output_dir: str, n_examples_to_train: int):
        '''
        Constructor of the class, receives the folder where the data is stored, the format of the columns in the data,
        the name of the train, test and dev files, the directory where the output will be stored, and an integer that
        defines how many lines the training is going to use.
        '''
        self._N_EXAMPLES_TO_TRAIN = n_examples_to_train
        self._data_foler: str = data_folder
        self._column_format: str = column_format
        self._train_file: str = train_file
        self._test_file: str = test_file
        self._dev_file: str = dev_file
        self._output_dir: str = output_dir
        self._embedding_types: list = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward-fast'),
            FlairEmbeddings('news-backward-fast'),
        ]
        self._embeddings = StackedEmbeddings(embeddings=self._embedding_types)
        self._model: SequenceTagger
        self._label_dictionary: Dictionary
        self._corpus: Corpus
        self._trainer: ModelTrainer
        self._plotter: Plotter


    def limit_lines(self):
        '''
        Function that creates temp files containing the number of lines specified to the class
        '''
        try:
            with open(os.path.join(self._data_foler, self._train_file), 'r') as f:
                with open('./data/train_slice', 'a') as t:
                    for i in range(self._N_EXAMPLES_TO_TRAIN):
                        line = f.readline()
                        t.write(line)
            
            with open(os.path.join(self._data_foler, self._test_file), 'r') as f:
                with open('./data/test_slice', 'a') as t:
                    for i in range(self._N_EXAMPLES_TO_TRAIN):
                        line = f.readline()
                        t.write(line)

            with open(os.path.join(self._data_foler, self._dev_file)) as f:
                with open('./data/dev_slice', 'a') as t:
                    for i in range(self._N_EXAMPLES_TO_TRAIN):
                        line = f.readline()
                        t.write(line)
            return True

        except Exception as e:
            return False
    

    def delete_temp_files(self):
        '''
        Function that deletes the created temp files
        '''
        try:
            os.remove('./data/train_slice')
            os.remove('./data/test_slice')
            os.remove('./data/dev_slice')
            return True

        except Exception as e:
            return False


    def get_corpus(self):
        '''
        Function that obtains the corpus and the label dictionary
        '''
        self._corpus = ColumnCorpus(data_folder=self._data_foler,
                                    column_format=self._column_format,
                                    train_file='train_slice',
                                    test_file='test_slice',
                                    dev_file='dev_slice')
        self._label_dictionary = self._corpus.make_label_dictionary(label_type='ner')
        return self._label_dictionary


    def create_model(self, hidden_size: int):
        '''
        Function that creates the ner model
        '''
        self._model = SequenceTagger(hidden_size=hidden_size,
                                     embeddings=self._embeddings,
                                     tag_dictionary=self._label_dictionary,
                                     tag_type='ner')
        return self._model

    
    def train(self, hidden_size: int, epochs: int = 5):
        '''
        Function that runs all the above steps plus the training of the model
        '''
        self.limit_lines()
        self.get_corpus()
        self.create_model(hidden_size=hidden_size)
        self._trainer = ModelTrainer(self._model, self._corpus)
        self._trainer.train(self._output_dir,
                            train_with_dev=True,
                            max_epochs=epochs,
                            train_with_test=True,
                            mini_batch_size=60)
        self.delete_temp_files()
        return

    def plot(self):
        '''
        Function that plots the loss of the training. The plot is stored as a png image in the
        output directory
        '''
        try:
            self._plotter = Plotter()
            self._plotter.plot_training_curves(os.path.join(self._output_dir,'loss.tsv'), ['loss'])
            return True

        except Exception as e:
            return False


# TESTS
if __name__ =='__main__':
    # ner = NER('./data',{0: 'text', 1: 'ner'},'train','test','dev','train_output', 3)
    # print(ner.get_corpus())
    # print(ner.create_model(256))
    # print(ner.limit_lines())
    # print(ner.delete_temp_files())

    # Max value of n_examples_to_train is 27, for some reason when the ner token B-other is found the code
    # dies. I found tried using add_unk=True but it didn't work either
    ner2 = NER('./data',{0: 'text', 1: 'ner'},'train','test','dev','train_output', 27)
    ner2.train(hidden_size=256, epochs=20)
    ner2.plot()

