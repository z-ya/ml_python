import collections
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
def tokenizer(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens
def cluster_sentences(texts, n=2) :
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'), lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=n)
    model.fit(matrix)
    topics = collections.defaultdict(list)

    for index, label in enumerate(model.labels_):
        topics[label].append(index)
    return dict(topics)

if __name__ =='__main__':
    sentences = ["Quantum physics is quite important in science nowadays.",
                 "Software engineering is hotter and hotter topic in the silicon valley",
                 "Investing in stocks and trading with them are not that easy",
                 "FOREX is the stock market for trading currencies",
                 "Warren Buffet is famous for making good investments. He knows stock markets"]
    n_clusters = 2
    clusters = cluster_sentences(sentences, n_clusters)

    for cluster in range(n_clusters):
        print('CLUSTER ', cluster, ":")
        for i, sentence in enumerate(clusters[cluster]):
            print('\tSENTERNCE ', i+1, ":", sentences[sentence])

