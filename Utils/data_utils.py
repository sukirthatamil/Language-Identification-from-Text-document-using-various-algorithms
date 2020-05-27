import os
import sys
import inspect
import nltk
dirpath = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

class Data:
	def __init__(self):
		"""Initializes Data Object for reading WiLI-2018 Dataset."""
		self.x_train_file = os.path.join(dirpath, '../Data/wili-2018/x_train.txt')
		self.y_train_file = os.path.join(dirpath, '../Data/wili-2018/y_train.txt')
		self.x_test_file = os.path.join(dirpath, '../Data/wili-2018/x_test.txt')
		self.y_test_file = os.path.join(dirpath, '../Data/wili-2018/y_test.txt')
		self.train = []
		self.test = []
		self.classes = set()
		self.read_data()
		
	def read_data(self):
		"""Reads the data into train and test categories from the datafiles."""
		self.main_train = []
		with open(self.x_train_file,'r') as xf, open(self.y_train_file,'r') as yf:
			for x,y in zip(xf, yf):
				# print(x,y)
				if not x.strip() == '':
					self.train.append((x.strip(),y.strip()))
					self.classes.add(y.strip())
		self.test = []
		with open(self.x_test_file,'r') as xf, open(self.y_test_file,'r') as yf:
			for x,y in zip(xf, yf):
				if not x.strip() == '':
					self.test.append((x.strip(),y.strip()))
					self.classes.add(y.strip())

	def data_iterator(self,segment='train', split_fraction = 0.8, y = None):
		"""Generator function which yields the data as per the given category (train, test, dev) and optionally from the given language."""
		assert segment in ['train','dev','test']
		self.split_fraction = split_fraction
		limit = int(self.split_fraction * len(self.train))
		data = None
		if segment=='train':
			data = self.train[:limit]
		elif segment=='dev':
			data = self.train[limit:]
		elif segment=='test':
			data = self.test
		
		if y is None:
			for s in data:
				yield s
		else:
			for s in data:
				if s[1]==y:	
					yield s

if __name__ == '__main__':
	d = Data()
	for language in d.classes:
		m = int(sys.argv[1])
		threshold = 50
		doc = ''
		for x,y in d.data_iterator(y = language):
			doc += x
		print('*'*5, language, '*'*5)
		for n in range(1,m+1):
			ngram_list = nltk.ngrams(doc,n)
			ngram_freq = nltk.FreqDist(ngram_list)
			for freq, ngram in sorted([(ngram_freq[k],k) for k in ngram_freq] ,reverse = True)[:threshold]:
				freq = ngram_freq[ngram]
				print(''.join(ngram), freq)
		print('\n\n\n\n\n')



























