import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import collections
import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
from Utils.data_utils import Data
from tqdm import tqdm
import pickle as pk
from collections import Counter
from math import log
import argparse
num_frequent_ngrams = 300 
langs = ['English', 'Spanish', 'French', 'Portuguese', 'German', 'Tamil', 'Assamese', 
            'Hindi', 'Turkish', 'Arabic']
labels_file = 'wili-2018/labels.csv'
x_train_file = 'wili-2018/x_train.txt'
y_train_file = 'wili-2018/y_train.txt'
x_test_file = 'wili-2018/x_test.txt'
y_test_file = 'wili-2018/y_test.txt'
labels_df = pd.read_csv(labels_file, sep=';')
lang_labels = list(labels_df[labels_df['English'].isin(langs)]['Label'])
lang_names = (list(labels_df[labels_df['English'].isin(langs)]['English']))
lf_df = pd.read_csv("ordered-letter-sequences.csv", skiprows = 0, sep=',')
lf_df.head(18)
root=Tk()
root.title("Language Identification")
def process_file(textfile):
    with open(textfile,encoding="utf8") as myfile:
        content = myfile.readlines()
    
    all_letters ='esaitnrulodcmpévqfbghjàxèyêzçôùâûîøöœwkäßïëüæñ'
    # initialize the dict with ordered entries for all letters, with each a value initialized to 0
    dic ={letter: 0 for letter in all_letters}
    total = 0
    for line in content:
        for letter in line:
            letter = letter.lower()
            if letter in all_letters:
                total += 1
                if letter in dic: dic[letter] += 1
                else: dic[letter] = 0

    # normalize
    for letter in dic:
        try:
            dic[letter] = dic[letter] / total
        except ZeroDivisionError:
            dic[letter]=0
            #mb.showinfo('leveinstein', "not found")
        
            
    if total is 0:
        mb.showinfo('leveinstein', "This method cannot detect this language..the upcoming one is default one")
    return dic
    
        
def read_file(x_file, y_file):
    
    
    # Read contents of 'y_file' into a dataframe
    y_df = pd.read_csv(y_file, header=None)
    # y_df has only one column; name it 'Label'
    y_df.columns = ['Label']

    # Read contents of 'x_file' into a list of strings
    with open(x_file, encoding='utf8') as f:
        x_pars = f.readlines()
    
   
    x_pars = [t.strip() for t in x_pars]
    # Convert the list into a dataframe, with one column: 'Par'
    x_df = pd.DataFrame(x_pars, columns=['Par']) 
    # Just keep paragraphs of languages in lang_labels (and remove other languages)
    x_df = x_df[y_df['Label'].isin(lang_labels)]
    # Just keep languages in lang_labels
    y_df = y_df[y_df['Label'].isin(lang_labels)]

    return (x_df, y_df)


def retrieve_input():
    inputValue=textBox.get("1.0","end-1c")
    print(inputValue)
    textfile=inputValue
    text_lf_dict = process_file(textfile)
    text_lf = pd.DataFrame.from_dict(text_lf_dict, orient='index', columns=['frequency'])
    text_lf['letter'] = text_lf.index
    text_lf.head(10)
    ''.join(text_lf[text_lf['frequency']>0].sort_values(by=['frequency'], ascending=False)['letter'])
    document_letter_sequence = ''.join(text_lf[text_lf['frequency']>0].sort_values(by=['frequency'], ascending=False)['letter'])
# loop over the letter sequences in lf_df - for each language, determine levenshtein distance with document_letter_sequence
    best_score = 999
    best_matching_language = None
    for index, row in lf_df.iterrows():
        ld = levenshtein(document_letter_sequence,row['ordered_letters'])
        print(row['language'],': ',ld)
        if ld < best_score:
            best_score= ld
            best_matching_language = row['language']
    #print("We have a winner: ",best_matching_language)
    mb.showinfo('leveinstein', best_matching_language)


def retrieve_input1():
    inputValue=textBox.get("1.0","end-1c")
    x_train_df, y_train_df = read_file(x_train_file, y_train_file)
    lang_pars = len(lang_labels)*[''] 
    for i in range(len(x_train_df)): # traverse rows of "x_train.txt" one by one
        lang_index = lang_labels.index(y_train_df['Label'].iloc[i]) # find index of language that this row belongs to
        lang_pars[lang_index] += ' ' + x_train_df.iloc[i].values[0] # concatinate this row to the string of the corresponding language
    lang_pars[0]
    arr=[]
    lang_profiles = []
    for i in range(len(lang_labels)):
        lang_profiles.append(most_frequent_ngrams(lang_pars[i]))
    str = open(inputValue, encoding="utf8").read()
    test_profile = most_frequent_ngrams(str)
    predicted_lang_label = lang_predictor(test_profile, lang_profiles)
    predicted_lang_label
    if predicted_lang_label in lang_labels:
    # Language name of the corresponding language label
        predicted_lang = lang_names[lang_labels.index(predicted_lang_label)]
    else: # Language name not found, return language label
        predicted_lang = predicted_lang_label
    print(predicted_lang)
    mb.showinfo('Canvar and Trenkle', predicted_lang)
def retrieve_input2():
    inputValue=textBox.get("1.0","end-1c")
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--n', help = 'n as in n-grams (default set to 3)', default = 3, type = int)
        parser.add_argument('--Lambda', help = 'pseudo count in additive smoothing. (default set to 1)', default = 1.0, type = float)
        parser.add_argument('--model_name', help = 'name of the model to save the parameters (default set to trigram)', default = 'new_bigram', type = str)
        parser.add_argument('--enable_train', help = 'to train the model and evaluate the model, otherwise it only evaluates the pretrained model.', action='store_true')
        parser.add_argument('--enable_language_restriction', help = 'To evaluate on selected 6 selected languages (German, French, English, Dutch, Italian, Spanish) instead all 235 languages', action='store_true')
        args, unknown= parser.parse_known_args()

    d = Data()
    m = Model(d, n = args.n, Lambda= args.Lambda, model_name = args.model_name)
    if args.enable_train:
        m.train()
    m.load_params()
    print('Evaluating...')
    language_set = None
    if args.enable_language_restriction:
        language_set = ['fra','eng','nld','spa','ita','deu']

  # print('Accuracy DEV set:',m.evaluate('dev', print_labels = True, language_set = language_set ))
  # print('Accuracy TEST set:',m.evaluate('test', print_labels = True, language_set = language_set ))
  #hindi_doc = 'विकिपीडिया सभी विषयों पर प्रामाणिक और उपयोग, परिवर्तन व पुनर्वितरण के लिए स्वतन्त्र ज्ञानकोश बनाने का एक बहुभाषीय प्रकल्प है। यह यथासम्भव निष्पक्ष दृष्टिकोण वाली सूचना प्रसारित करने के लिए कृतसंकल्प है। सर्वप्रथम अंग्रेज़ी विकिपीडिया जनवरी 2001 में आरम्भ किया गया '
    doc = open(inputValue,encoding="utf8").read()
  #eng_doc='HI It is English. It is very important'
    answer=m.predict(doc)
    mb.showinfo('bayesian',answer)
textBox=Text(root, height=2, width=25)
textBox.pack()
buttonbayes=Button(root, height=1, width=30, text="Bayesian", 
                    command=lambda: retrieve_input2())

#command=lambda: retrieve_input() >>> just means do this when i press the button
buttonbayes.pack()
buttonCommit=Button(root, height=1, width=30, text="Canvar and Trenkle", 
                    command=lambda: retrieve_input1())

#command=lambda: retrieve_input() >>> just means do this when i press the button
buttonCommit.pack()
buttonleveinstein=Button(root, height=1, width=10, text="leveinstein", 
                    command=lambda: retrieve_input())
#command=lambda: retrieve_input() >>> just means do this when i press the button
buttonleveinstein.pack()
T = tk.Text(root, height=4, width=50)

T.pack(side=tk.LEFT, fill=tk.Y)
quote = """Some examples of different languages:
English: Today we are demonstrating an automatic language identification system using machine learning.
French: Comment ça va aujourd'hui?
German: Heute demonstrieren wir ein automatisches Spracherkennungssystem mit maschinellem Lernen.
Hindi: आज हम मशीन लर्निंग का उपयोग करके एक स्वचालित भाषा पहचान प्रणाली का प्रदर्शन कर रहे हैं।
Polish: Dziś demonstrujemy automatyczny system identyfikacji języka za pomocą uczenia maszynowego.

Tamil: இயந்திர கற்றலைப் பயன்படுத்தி தானியங்கி மொழி அடையாள முறையை இன்று நாங்கள் நிரூபிக்கிறோம்.
Turkish: Bugün makine öğrenmesini kullanan bir otomatik dil tanımlama sistemi gösteriyoruz."""
T.insert(tk.END, quote)

mainloop()
class Model:
	def __init__(self, data, n = 3, Lambda = 1.0, model_name = 'new_trigram'):
		"""Initializes an ngram naive bayes model with Add One Smoothing by default."""
		self.n = n
		self.model_name = model_name
		self.save_path = 'Params/model_{}'.format(self.model_name)
		self.Lambda = Lambda
		self.data = data
		self.prior = dict([(lang,0.0) for lang in self.data.classes])
		self.N = float(len(list(self.data.data_iterator('train'))))
		self.language_specific_ngrams_total = dict()
		self.ngram_count = dict()

	def get_ngrams(self, x):
		"""Returns a list of character ngrams for the given document(string)."""
		ngram_list = []
		for i in range(len(x)):
			for n in range(self.n):
				if i >= n:
					gm = x[i-n:i+1]
					ngram_list.append(gm)
		return ngram_list

	def update_language_ngrams(self, x, y):
		"""Updates count of ngrams for the specified language."""
		ngrams_list = self.get_ngrams(x)
		self.language_specific_ngrams_total[y] = len(ngrams_list)
		self.ngram_count[y] = Counter(ngrams_list)

	def train(self):
		"""Trains the model, by updating the counts for the given dataset."""
		print('training ...')
		for language in tqdm(self.data.classes):
			doc = ''
			n_examples = 0 
			for x,y in self.data.data_iterator('train',y = language):
				doc += x
				n_examples += 1
			self.prior[language] = n_examples/self.N
			self.update_language_ngrams(doc, language)
		self.save_params()

	def get_probability(self, y, ngram_list):
		"""Returns unnormalised probability for a given ngram_list and language."""
		p = 0.0
		for ngram in ngram_list:
			likelihood = (self.ngram_count[y][ngram] + self.Lambda )/ ( self.language_specific_ngrams_total[y] + self.Lambda*len(self.ngram_count[y]) )
			log_likelihood = log( likelihood )
			p += log_likelihood
		p = p+log(self.prior[y])
		return p

	def predict(self, doc, language_set=None):
		"""Returns the predicted label for the docment."""
		if language_set is None:
			language_set = list(self.data.classes)
		doc = doc.strip()
		ngram_list = self.get_ngrams(doc)
		probabilities = []
		for language in language_set:
			p = self.get_probability(language, ngram_list)
			probabilities.append( p )
		return language_set[np.argmax(probabilities)]

	def save_params(self):
		"""Saves the parameters of the model"""
		print('saving ...')
		params = [self.prior, self.language_specific_ngrams_total, self.ngram_count]		
		with open(self.save_path,'wb') as f:
			pk.dump(params, f)

	def load_params(self):
		"""Loads the parameters of the model"""
		print('loading ...')
		with open(self.save_path,'rb') as f:
			self.prior, self.language_specific_ngrams_total, self.ngram_count = pk.load(f)

	def evaluate(self, segment='train', print_labels = False, language_set = None):
		"""Evaluates the model and returns the accuracy score. Keep languege_set None to evaluate on all the 235 languages"""
		assert segment in ['train','test','dev']
		print('evaluating {} set...'.format(segment))
		total, correct = 0.0, 0.0
		if language_set is None:
			language_set = list(self.data.classes)
		language_allowed = dict([(l,True) if l in language_set else (l,False) for l in list(self.data.classes)])
		for i,ex in enumerate(self.data.data_iterator(segment)):
			x,y = ex
			if language_allowed[y]:
				y_predicted = '-'
				y_predicted = self.predict(x.strip(), language_set)
				total += 1
				if print_labels:
					print(i, y_predicted, y)
				if y_predicted == y:
					correct += 1
		return correct/total

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    # print (matrix)
    return (matrix[size_x - 1, size_y - 1])

def getNgrams(paragraph, n):
    
    ngrams = []
    for _n in range(1,n+1):
        _par_padded = ' ' + paragraph.lower() + ' '  #  ngrams at the edges of paragraphs are padded with space.
        for pos in range(0,len(_par_padded)-_n+1):
            _ngram = _par_padded[pos:pos+_n]
            if ' ' not in _ngram[1:-1]: # ngrams with inside spaces (ngrams formed from two words) are not considered
                ngrams.append(_ngram)
            #print(count(_ngram))
    return ngrams
def most_frequent_ngrams(par):
   
    ngrams = getNgrams(par, 5) # ngram range: 1 to 5
    freqs = collections.Counter(ngrams).most_common(num_frequent_ngrams)
    return [ngram for (ngram, fr) in freqs]

def lang_predictor(test_profile, lang_profiles):
    """
    Using language profiles learned from train files, predict the language of test_profile.
    The function computes distance of test_profile from each train language profile and
    return language with minimum distance.

    @param test_profile: language profile of a test paragraph
    @type test_profile: list

    @param lang_profiles: list of previously learned language profiles
    @type lang_profiles: list

    @return: label of the predicted language
    """
    # Compute distance of test_profile from each train language profile.
    # Distance criteria: Canvar-Trenkle distance
    distances = []
    for pr in lang_profiles:
        pr_distance = 0
        for n_gram in test_profile:
            if n_gram in pr:
                # Determine how far out of place an n-gram in test_profile is from its place in each of lang_profiles.
                d = list(pr).index(n_gram) - list(test_profile).index(n_gram)
            else:
                # n-gram is not in any of lang_profiles, so it takes maximum out-of-place distance (num_frequent_ngrams=300)
                d = num_frequent_ngrams
            # distance: sum of all of the out-of-place values for all n-grams
            pr_distance += d

        distances.append(pr_distance)
    # return label of language with minimum distance 
    return lang_labels[np.argmin(distances)]

