import nltk
import os
import sys

folder = 'docs/'
cluster_range = (1, 4)
lemmatize = True
rm_stopwords = True
num_sentences = 5
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


def get_clusters(folder):
	# Store dict of lists of file paths, one list per cluster
	clusters = {}
	# Initialize clusters in dict
	for i in range(cluster_range[0], cluster_range[1]+1):
		clusters[i] = []
	# Populate cluster dict
	for f in os.listdir(folder):
		for i in range(cluster_range[0], cluster_range[1]+1):
			if f.startswith('doc%d' % i):
				clusters[i].append(os.path.join(folder, f))
				break
	return clusters


def get_probabilities(cluster, lemmatize, rm_stopwords):
	# Store word probabilities for this cluster
	word_ps = {}
	# Keep track of the number of tokens to calculate probabilities later
	token_count = 0.0
	# Gather counts for all words in all documents
	for path in cluster:
		with open(path) as f:
			tokens = clean_sentence(nltk.word_tokenize(f.read()))
			token_count += len(tokens)
			for token in tokens:
				if token not in word_ps:
					word_ps[token] = 1.0
				else:
					word_ps[token] += 1.0
	# Divide word counts by the number of tokens across all files
	for word_p in word_ps:
		word_ps[word_p] = word_ps[word_p]/float(token_count)
	return word_ps


def get_sentences(cluster):
	sentences = []
	for path in cluster:
		with open(path) as f:
			sentences += nltk.sent_tokenize(f.read())
	return sentences


def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize: tokens = [lemmatizer.lemmatize(t) for t in tokens]
	if rm_stopwords: tokens = [t for t in tokens if t not in stopwords]
	return tokens


def score_sentence(sentence, word_ps):
	score = 0.0
	num_tokens = 0.0
	sentence = nltk.word_tokenize(sentence)
	tokens = clean_sentence(sentence)
	for token in tokens:
		if token in word_ps:
			score += word_ps[token]
			num_tokens += 1.0
	return float(score)/float(num_tokens)


def max_sentence(sentences, word_ps, simplified):
	max_sentence = None
	max_score = None
	for sentence in sentences:
		score = score_sentence(sentence, word_ps)
		if score > max_score or max_score == None:
			max_sentence = sentence
			max_score = score
	if not simplified: update_ps(max_sentence, word_ps)
	return max_sentence


def update_ps(max_sentence, word_ps):
	sentence = nltk.word_tokenize(max_sentence)
	sentence = clean_sentence(sentence)
	for word in sentence:
		word_ps[word] = word_ps[word]**2
	return True


def sumbasic(clusters, simplified):
	summaries = {}
	for cluster in clusters:
		summaries[cluster] = []
		# Get word probabilities for each word
		word_ps = get_probabilities(clusters[cluster], lemmatize, rm_stopwords)
		# Get all the sentences in a cluster
		sentences = get_sentences(clusters[cluster])
		# Compile summary
		for i in range(num_sentences):
			summaries[cluster].append(max_sentence(sentences, word_ps, simplified))
	return summaries


def leading(clusters):
	summaries = {}
	for cluster in clusters:
		summaries[cluster] = []
		sentences = get_sentences(clusters[cluster])
		for i in range(num_sentences):
			summaries[cluster].append(sentences[i])
	return summaries

 def main():
	# Get dict of lists, one list for each cluster
	print "Getting clusters..."
	clusters = get_clusters(folder)
	print "Simple sumbasic..."
	simple = sumbasic(clusters, True)
	print "Original sumbasic..."
	orig = sumbasic(clusters, False)
	print "Leading..."
	lead = leading(clusters)
	for cluster in clusters:
		print "##########################################"
		print "Cluster: " + str(cluster)
		print ""
		print "Simple: " + " ".join(simple[cluster])
		print ""
		print "Original: " + " ".join(orig[cluster])
		print ""
		print "Leading: " + " ".join(lead[cluster]) 

def main():
	method = sys.argv[1]
	cluster = sys.argv[2]
	summary = exec method(cluster)
	print summary

if __name__ == '__main__':
	main()
