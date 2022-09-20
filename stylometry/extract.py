from __future__ import division
import glob
import re
import os
import nltk
nltk.download('punkt')

from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import numpy as np
import random

DEFAULT_AUTHOR = "Unknown"
n_lines = 0
crea_path = "C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/stylometry/data/crea_total.txt"

#C:/Users/Cesar/Documents/GitHub/stylometry-master3/stylometry/data

class StyloDocument(object):

    def __init__(self, file_name, author=DEFAULT_AUTHOR):
        self.doc = open(file_name, "r", encoding='utf-8').read()
        #Separar signos y palabras con expresion regular
        self.separateWordsPuntation = re.findall(r"[\w']+|[.,!?;:-“”()<>\[\]\´\´\"\"\{\}\-\+\*\/]", self.doc) 
        self.author = author
        self.file_name = file_name
        self.tokens = word_tokenize(self.doc)
        self.text = Text(self.separateWordsPuntation)
        self.fdist = FreqDist(self.text)
        self.sentences = sent_tokenize(self.doc)
        self.sentence_chars = [ len(sent) for sent in self.sentences]
        self.sentence_word_length = [ len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.doc.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]

    @classmethod
    def csv_header(cls):
        words = cls.word_extraction(cls, crea_path, n_lines)

        header = ['Author','Title','LexicalDiversity','MeanWordLen',
        'MeanSentenceLen','StdevSentenceLen','MeanParagraphLen','DocumentLen',
        'Commas','Puntos','Guion',') Paréntesis',') Paréntesis','Dos puntos','. Comillas','¿ Interrogación','? Interrogación','Punto y coma']

        for word in words:
            header.append(word)
            
        return (",".join(header))

    def word_extraction(self, path, n_lines):
        words_pattern = '[a-z\u00C0-\u017F]+|[\;]' # expresion regular
        myfile = open(path, "r", encoding='utf-8')

        result = []

        # omitir primera linea
        next(myfile)

        for i in range(n_lines):
            line = myfile.readline()
            # filtra solo las palabras
            realLine = re.findall(words_pattern, line, flags=re.IGNORECASE)
            result.append(realLine[0])

        return result

    def term_per_thousand(self, term):
        """
        term       X
        -----  = ------
          N       1000
        """
        return (self.fdist[term] * 1000) / self.fdist.N()

    def mean_sentence_len(self):
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        return np.mean(self.paragraph_word_length)
        
    def std_paragraph_len(self):
        return np.std(self.paragraph_word_length)

    def mean_word_len(self):
        words = set(word_tokenize(self.doc))
        word_chars = [ len(word) for word in words]
        return sum(word_chars) /  float(len(word_chars))

    def lengths(self):
        return [
            self.author, 
            self.file_name, 
            self.type_token_ratio(), 
            self.mean_word_len(), 
            self.mean_sentence_len(),
            self.std_sentence_len(),
            self.mean_paragraph_len(), 
            self.document_len()
        ]

    def punctuation_marks(self):
        return [
            self.term_per_thousand(','),
            self.term_per_thousand('.'),
            self.term_per_thousand('-'),
            self.term_per_thousand(')'),
            self.term_per_thousand('('),
            self.term_per_thousand(':'),
            self.term_per_thousand('"'),
            self.term_per_thousand('?'),
            self.term_per_thousand('¿'),
            self.term_per_thousand(';'),
        ]

    def space_between(self, count, outfile):
        path = 'C:/Users/crisc/OneDrive/Desktop/pruebas_tesis/stylometry-master3/pan15-verification-training-sp/'
        dirname = ''
        text_merged = ''
        text_unknown = ''

        if count < 10:
            dirname = 'SP00'+str(count)
        elif count < 100:
            dirname = 'SP0'+str(count)
        else:
            dirname = 'SP'+str(count)

        if outfile == 'novels.csv':
            path = path+dirname+'/merged.txt'
            with open(path,encoding='utf-8') as fl:
                text_merged += fl.read() + '\n'
            with open(path, 'w', encoding='utf-8') as fl:
                fl.write(text_merged)

        else:
            path = path+dirname+'/unknown.txt'
            with open(path,encoding='utf-8') as fl:
                text_unknown += fl.read() + '\n'
            with open(path, 'w', encoding='utf-8') as fl:
                fl.write(text_unknown)

    def type_token_ratio(self):
        return (len(set(self.text)) / len(self.text)) * 100

    def unique_words_per_thousand(self):
        return self.type_token_ratio()/100.0*1000.0 / len(self.text)

    def document_len(self):
        return sum(self.sentence_chars)

    def csv_output(self, count, outfile):
        words = self.word_extraction(crea_path, n_lines)
        # str = '"%s","%s",%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g'
        first = self.lengths()

        second = self.punctuation_marks()

        my_list = first + second
        for word in words:
            my_list.append(self.term_per_thousand(word))

        tpl = tuple(my_list)
        strng = ",".join(map(str, tpl))
        
        return strng

    
    def text_output(self):
        print("##############################################")
        print("")
        print("Name: ", self.file_name)
        print("Author: ", self.author)
        print("")
        print(">>> Phraseology Analysis <<<")
        print("")
        print("Lexical diversity        :", self.type_token_ratio())
        print("Mean Word Length         :", self.mean_word_len())
        print("Mean Sentence Length     :", self.mean_sentence_len())
        print("STDEV Sentence Length    :", self.std_sentence_len())
        print("Mean paragraph Length    :", self.mean_paragraph_len())
        print("Document Length          :", self.document_len())
        print("")
        print(">>> Punctuation Analysis (per 1000 tokens) <<<")
        print("")
        print('Commas                   :', self.term_per_thousand(','))
        print('Semicolons               :', self.term_per_thousand(';'))
        print('Quotations               :', self.term_per_thousand('\"'))
        print('Exclamations             :', self.term_per_thousand('!'))
        print('Colons                   :', self.term_per_thousand(':'))
        print('Hyphens                  :', self.term_per_thousand('-')) # m-dash or n-dash?
        print('Double Hyphens           :', self.term_per_thousand('--')) # m-dash or n-dash?
        print("")
        print(">>> Lexical Usage Analysis (per 1000 tokens) <<<")
        print("")
        
                
        words = self.word_extraction(crea_path, n_lines)
        
        for x in words:
            print('%s                      :%g'%(x, self.term_per_thousand(x)))



class StyloCorpus(object):

    
    def __init__(self,documents_by_author):
        self.documents_by_author = documents_by_author

    @classmethod
    def from_path_list(cls, path_list, author=DEFAULT_AUTHOR):
        stylodoc_list = cls.convert_paths_to_stylodocs(path_list)
        documents_by_author = {author:stylodoc_list}
        return cls(documents_by_author)

    @classmethod
    def from_stylodoc_list(cls, stylodoc_list, author=DEFAULT_AUTHOR):
        author = DEFAULT_AUTHOR
        documents_by_author = {author:stylodoc_list}
        return cls(documents_by_author)

    @classmethod
    def from_documents_by_author(cls, documents_by_author):
        return cls(documents_by_author)

    @classmethod
    def from_paths_by_author(cls, paths_by_author):
        documents_by_author = {}
        for author, path_list in paths_by_author.iteritems():
            documents_by_author[author] = cls.convert_paths_to_stylodocs(path_list,author)
        return cls(documents_by_author)

    @classmethod
    def from_glob_pattern(cls, pattern, n_words):
        global n_lines
        n_lines = n_words
        
        documents_by_author = {}
        if isinstance(pattern,list):
            for p in pattern:
                documents_by_author.update(cls.get_dictionary_from_glob(p))
        else:
            documents_by_author = cls.get_dictionary_from_glob(pattern)
        return cls(documents_by_author)

    @classmethod
    def convert_paths_to_stylodocs(cls, path_list, author=DEFAULT_AUTHOR):
        stylodoc_list = []
        for path in path_list:
            sd = StyloDocument(path, author)
            stylodoc_list.append(sd)
        return stylodoc_list

    @classmethod
    def get_dictionary_from_glob(cls, pattern):
        documents_by_author = {}
        for path in glob.glob(pattern):
            author = path.split('\\')[-2]
            document = StyloDocument(path, author)
            if author not in documents_by_author:
                documents_by_author[author] = [document]
            else:
                documents_by_author[author].append(document)
        return documents_by_author

    def output_csv(self, out_file, author=None):
        print(out_file)
        csv_data = StyloDocument.csv_header() + '\n'
        count = 0
        if not author:
            for a in self.documents_by_author.keys():
                for doc in self.documents_by_author[a]:
                    count += 1
                    csv_data += doc.csv_output(count, os.path.basename(out_file)) + '\n'
        else:
            for doc in self.documents_by_author[author]:
                count += 1
                csv_data += doc.csv_output(count, os.path.basename(out_file)) + '\n'
        if out_file:
            with open(out_file,'w') as f:
                f.write(csv_data)
        return csv_data
            