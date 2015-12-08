#!/usr/bin python
import json
import pandas
import numpy
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def split_into_tokens(recipe):
	return TextBlob(recipe).words

def convert_to_lemmas(recipe):
	ingredients = ' '.join(recipe)
	print(ingredients)
	words = TextBlob(ingredients).words 
	print(type(words[1]))
	return [word.lemma() for word in words]

with open('Data/train.json/train.json') as f:
	data = f.read()
	jsondata = json.loads(data)
	
jsondata = pandas.read_json('Data/train.json/train.json')
jsondata['length'] = jsondata['ingredients'].map(lambda text: len(text))

bow_transformer = CountVectorizer(analyzer = convert_to_lemmas).fit(jsondata['ingredients'])
print len(bow_transformer.vocabulary_)
print bow_transformer.vocabulary_
