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
        # https://towardsdatascience.com/whats-the-meaning-of-single-and-double-underscores-in-python-3d27d57d6bd1
        # single underscore not double! 
        self.__source_file = source_file
        self.__target_file = target_file
        self.__deepl_key = deepl_key
        self.__deepl_bleu = 0
        self.__google_bleu = 0


    def deepl_translation(self):
        '''
        Runs the translation with Deepl's translator and computes the average bleu of the
        translations
        '''
        translator = deepl.Translator(self.__deepl_key)
        num_lines = 0
        with open(self.__source_file, 'r', encoding='utf-8') as s: # single-letter variable names are discouraged! 
            with open(self.__target_file, 'r', encoding='utf-8') as t:
                for line in s:
                    ref = [t.readline().split()]
                    translation = translator.translate_text(line, source_lang='ES', target_lang='EN-US')
                    result = str(translation).split()
                    self.__deepl_bleu += sentence_bleu(ref, result, (1, 0, 0, 0))
                    num_lines += 1
                    
        # when you use "with open(", it closes the file for you! https://www.programiz.com/python-programming/file-operation
        print(f'DEEPL_TRANSLATOR: {self.__deepl_bleu/n}')


    def google_translation(self):
        '''
        Runs the translation with Google's translator and computes the average bleu of the
        translations
        '''
        # same changes from above apply here! 
        # you could refactor to have a function that gets the translations as a list of texts, and then another that computes the
        # bleu score that gets called in both places, to avoid repeating so much code. 
        translator = Translator()
        n = 0
        with open(self.__source_file, 'r', encoding='utf-8') as s:
            with open(self.__target_file, 'r', encoding='utf-8') as t:
                for line in s:
                    ref = [t.readline().split()]
                    translation = translator.translate(line, dest='en').text
                    result = translation.split()
                    self.__google_bleu += sentence_bleu(ref, result, (1, 0, 0, 0))
                    n += 1
                t.close()
            s.close()
        print(f'GOOGLE_TRANSLATOR: {self.__google_bleu/n}')

        
# TESTS
if __name__ == '__main__':
    t = Translations('spanish.txt', 'english.txt', 'ad448610-53c2-88aa-0b42-6371aef5f3aa:fx')
    t.deepl_translation()
    t.google_translation()
