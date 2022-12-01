import deepl
from nltk.translate.bleu_score import sentence_bleu
from googletrans import Translator

class Translations:
    '''
    Class that solves the third problem
    '''

    def __init__(self, source_file, target_file, deepl_key):
        '''
        Constructor of the class, receives the file to be translated, the file to evaluate
        the bleu of the translation and the Deepl API key
        '''
        self._source_file = source_file
        self._target_file = target_file
        self._deepl_key = deepl_key
        self._deepl_bleu = 0
        self._google_bleu = 0


    def deepl_translation(self):
        '''
        Runs the translation with Deepl's translator and computes the average bleu of the
        translations
        '''
        translator = deepl.Translator(self._deepl_key)
        num_lines = 0
        with open(self._source_file, 'r', encoding='utf-8') as s:
            with open(self._target_file, 'r', encoding='utf-8') as t:
                for line in s:
                    ref = [t.readline().split()]
                    translation = translator.translate_text(line, source_lang='ES', target_lang='EN-US')
                    result = str(translation).split()
                    self._deepl_bleu += sentence_bleu(ref, result, (1, 0, 0, 0))
                    num_lines += 1

        print(f'DEEPL_TRANSLATOR: {self._deepl_bleu/num_lines}')


    def google_translation(self):
        '''
        Runs the translation with Google's translator and computes the average bleu of the
        translations
        '''
        translator = Translator()
        num_lines = 0
        with open(self._source_file, 'r', encoding='utf-8') as s:
            with open(self._target_file, 'r', encoding='utf-8') as t:
                for line in s:
                    ref = [t.readline().split()]
                    translation = translator.translate(line, dest='en').text
                    result = translation.split()
                    self._google_bleu += sentence_bleu(ref, result, (1, 0, 0, 0))
                    num_lines += 1

        print(f'GOOGLE_TRANSLATOR: {self._google_bleu/num_lines}')

        
# TESTS
if __name__ == '__main__':
    t = Translations('spanish.txt', 'english.txt', 'ad448610-53c2-88aa-0b42-6371aef5f3aa:fx')
    t.deepl_translation()
    t.google_translation()