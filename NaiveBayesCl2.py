from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset="train", categories=categories, shuffle=True)
#print('\n'.join(training_data.data[10].split('\n')[:10]))
#print('target is:', training_data.target_names[training_data.target[10]])

count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
print(count_vector.vocabulary_)

tfid_transformer = TfidfTransformer()
x_train_tfidf = tfid_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)

