#!/usr/bin/python
# encoding=utf-8
from subprocess import call, STDOUT
from glob import glob
import nltk
from nltk.corpus import stopwords
import os, struct
from tensorflow.core.example import example_pb2
import pyrouge
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import *
from ast import literal_eval
import time
stemmer = PorterStemmer()


DATAPATH = '../../../data/test'
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
TARGET = 'RSAsummarization'

DATA_TO_TEST = {
    'future': 'future/test.txt.oracle',
    'contribution': 'contribution/test.txt.oracle',
    'baseline': 'baseline/test.txt.oracle',
    'dataset': 'dataset/test.txt.oracle',
}

Question = {
    'future': 'What is the future work of this paper?',
    'contribution': 'What are the contributions of this paper?',
    'baseline': 'Which baselines are compared with our model?',
    'dataset': 'What are the datasets in the experiements?',
}

TEST_PAPER = {key: value.replace('oracle', 'src') for key, value in DATA_TO_TEST.items()}
PAPER_REF = {key: value.replace('oracle', 'ref') for key, value in DATA_TO_TEST.items()}


# 4000 is too short and will cause error in batcher.py during initialization
# I guess it means the maximum "words" in a single paper. If a paper has 500 sentences...
# In run_summarization.py: tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
ratio = 0.2
max_enc = 800

# Shorten the document with ratio to reduce the max_enc length
# ratio = 0.4
# max_enc = 4000
# ratio = 0.2
# max_enc = 2000

cmd = 'python2 run_summarization.py --mode=decode --single_pass=1 --coverage=True --vocab_path=finished_files/vocab --log_root=. --exp_name=pretrained_model_tf1.2.1 --data_path=test/temp_file/{} --max_enc_steps={}'

generated_path = 'pretrained_model_tf1.2.1/decode_test_{}maxenc_4beam_35mindec_100maxdec_ckpt-238410'.format(max_enc)
stopwords = set(stopwords.words('english'))

max_len = 250


VERBOSE = False # if not VERBOSE, then it is always override mode
if not VERBOSE:
    log_file_name = 'test/call_log' + str(time.time()) + '.log'
    # https://stackoverflow.com/questions/11269575/how-to-hide-output-of-subprocess-in-python-2-7
    CallLog = open(log_file_name, 'w')
    print 'Since you are not in VERBOSE mode. Use tail -f', log_file_name, 'to track the log of the subprocesses.'


def pp(string):
    return ' '.join([stemmer.stem(word.decode('utf8')) for word in string.lower().split() if not word in stopwords])
    
def write_to_file(article, abstract, rel, writer):
    abstract = '<s> '+' '.join(abstract)+' </s>'
    #abstract = abstract.encode('utf8', 'ignore')
    #rel = rel.encode('utf8', 'ignore')
    #article = article.encode('utf8', 'ignore')
    tf_example = example_pb2.Example()
    tf_example.features.feature['abstract'].bytes_list.value.extend([bytes(abstract)])
    tf_example.features.feature['relevancy'].bytes_list.value.extend([bytes(rel)])
    tf_example.features.feature['article'].bytes_list.value.extend([bytes(article)])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))


# def duck_iterator(i):
#     duc_folder = 'duc0' + str(i) + 'tokenized/'
#     for topic in os.listdir(duc_folder + 'testdata/docs/'):
#         topic_folder = duc_folder + 'testdata/docs/' + topic
#         if not os.path.isdir(topic_folder):
#             continue
#         query = ' '.join(open(duc_folder + 'queries/' + topic).readlines())
#         model_files = glob(duc_folder + 'models/' + topic[:-1].upper() + '.*')

#         topic_texts = [' '.join(open(topic_folder + '/' + file).readlines()).replace('\n', '') for file in
#                        os.listdir(topic_folder)]

#         abstracts = [' '.join(open(f).readlines()) for f in model_files]
#         yield topic_texts, abstracts, query


def paper_iterator(topic):
    """ seems abstracts didn't used """
    src_file = os.path.join(DATAPATH, TEST_PAPER[topic])
    query = Question[topic]

    for paper in open(src_file, 'r').readlines():
        # topic_texts = [' '.join(paper.strip().split('##SENT##'))]
        topic_texts = [sent.strip() for sent in paper.split('##SENT##')]

        yield topic_texts, query
    

def ones(sent, ref): return 1.

def count_score(sent, ref):
    # ref = pp(ref).split()
    ref = pp(ref).split()
    sent = ' '.join(pp(w) for w in sent.lower().split() if not w in stopwords)
    return sum([1. if w in ref else 0. for w in sent.split()])


def get_w2v_score_func(magic = 10):
    import gensim
    google = gensim.models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)
    def w2v_score(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        sent = [w for w in sent.split() if w in google]
        ref = [w for w in ref.split() if w in google]
        try:
            score = google.n_similarity(sent, ref)
        except:
            score = 0.
        return score * magic
    return w2v_score

# def get_tfidf_score_func_glob(magic = 1):
#     corpus = []
#     for i in range(5, 8):
#         for topic_texts, _, _ in duck_iterator(i):
#             corpus += [pp(t) for t in topic_texts]

#     vectorizer = TfidfVectorizer()
#     vectorizer.fit_transform(corpus)

#     def tfidf_score_func(sent, ref):
#         #ref = [pp(s) for s in ref.split(' . ')]
#         sent = pp(sent)
#         v1 = vectorizer.transform([sent])
#         #v2s = [vectorizer.transform([r]) for r in ref]
#         #return max([cosine_similarity(v1, v2)[0][0] for v2 in v2s])
#         v2 = vectorizer.transform([ref])
#         return cosine_similarity(v1, v2)[0][0]

#     return tfidf_score_func

# tfidf_score = get_tfidf_score_func_glob()


def get_tfidf_score_func(topic, magic = 10):
    corpus = []
    for topic_texts, _ in paper_iterator(topic):
        corpus += [t.lower() for t in topic_texts]

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)

    def tfidf_score_func(sent, ref):
        ref = ref.lower()
        sent = sent.lower()
        v1 = vectorizer.transform([sent])
        v2 = vectorizer.transform([ref])
        return cosine_similarity(v1, v2)[0][0]*magic

    return tfidf_score_func


def just_relevant(sents, query, score_func):
    score_per_sent = [count_score(sent, query) for sent in sents]
    sents_gold = list(zip(*sorted(zip(score_per_sent, sents), reverse=True)))[1]
    sents_gold = sents_gold[:int(len(sents_gold)*ratio)]

    filtered_sents = []
    for s in sents:
        if not s: continue
        if s in sents_gold: filtered_sents.append(s)
    return filtered_sents

class Summary:
    def __init__(self, texts, abstracts, query, score_func):
        #texts = sorted([(tfidf_score(query, text), text) for text in texts], reverse=True)
        #texts = sorted([(tfidf_score(text, ' '.join(abstracts)), text) for text in texts], reverse=True)

        #texts = [text[1] for text in texts]
        self.texts = texts
        self.abstracts = abstracts
        self.query = query
        self.score_func = score_func
        self.summary = []
        self.words = set()
        self.length = 0

        self.sent_num = 0
        self.word_num = 0

    def add_sum(self, summ):
        for sent in summ:
            self.summary.append(sent.strip())

    def get(self):
        # text = max([(len(t.split()), t) for t in  self.texts])[1]
        #text = texts[0]
        # if ratio < 1: text = just_relevant(text, self.query)

        sents = self.texts

        if ratio < 1:
            sents = just_relevant(sents, self.query, self.score_func)

        
        self.sent_num = len(sents)
        text = ' '.join(sents)
        self.word_num = len(text.split())

        score_per_sent = [(self.score_func(sent, self.query), sent) for sent in sents]
        #score_per_sent = [(count_score(sent, ' '.join(self.abstracts)), sent) for sent in sents]

        scores = []
        for score, sent in score_per_sent:
            scores += [score] * (len(sent.split()))
        # make sure "word count" is the same as "scores" (one score match one word)
        assert len(text.split()) == len(scores)
        scores = str(scores)
        return text, 'a', scores

def get_summaries(path):
    path = os.path.join(path, 'decoded')
    out = {}
    for file_name in os.listdir(path):
        index = int(file_name.split('_')[0])
        out[index] = open(os.path.join(path, file_name)).readlines()
    return out

def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference_(\d+).txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    return r.convert_and_evaluate()

def calculate_performance(predicts, golds):
    total_gold_positive, total_predicted_positive = 0, 0
    total_hit1, total_correct = 0, 0
    for i, (os, ts) in tqdm(enumerate(zip(predicts, golds)), total=len(golds)):
        os = set(os)
        ts = set(ts)

        correct = os & ts
        total_correct += len(correct)
        if len(correct) > 0:
            total_hit1 += 1
        only_in_predict = os - ts
        only_in_annotation = ts - os

        total_gold_positive += len(ts)
        total_predicted_positive += len(os)
        precision = total_correct / total_predicted_positive
        recall = total_correct / total_gold_positive
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

    return {
        'acc_hit1': total_hit1 / len(golds),
        'p': precision,
        'r (acc_sentence_level)': recall,
        'f1': f1
    }


def compute_bleu_raw(gold_sents, predict_sents, gold=None, predict=None):
    avg_sent_bleu = 0
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    all_references = []
    all_hypothesis = []

    if gold and predict:
        for gs, ps, gid, pid in zip(gold_sents, predict_sents, gold, predict):

            gs = [sent for sent, _ in sorted(zip(gs, gid), key=lambda pair: pair[1])]
            ps = [sent for sent, _ in sorted(zip(ps, pid), key=lambda pair: pair[1])]

            references = [sent.split() for sent in gs]
            hypothesis = [word for sent in ps for word in sent.split()]

            avg_sent_bleu += nltk.translate.bleu_score.sentence_bleu(references, hypothesis,
                                                            smoothing_function=chencherry.method1)
            
            all_references.append(references)
            all_hypothesis.append(hypothesis)

        corpus_bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis)
        avg_sent_bleu /= len(gold_sents)
    else:
        for gs, ps in zip(gold_sents, predict_sents):
            references = [sent.split() for sent in gs]
            hypothesis = [word for sent in ps for word in sent.split()]

            avg_sent_bleu += nltk.translate.bleu_score.sentence_bleu(references, hypothesis,
                                                            smoothing_function=chencherry.method1)
            
            all_references.append(references)
            all_hypothesis.append(hypothesis)

        corpus_bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis,
                                                    smoothing_function=chencherry.method1)
        avg_sent_bleu /= len(gold_sents)

    return {'corpus_bleu': corpus_bleu, 'avg_sent_bleu': avg_sent_bleu}



def evaluate(summaries, topic):
    golds, predicts, gold_sents, predict_sents = [], [], [], []

    src_file = os.path.join(DATAPATH, TEST_PAPER[topic])
    with open(src_file, 'r') as stream:
        raw_papers = stream.readlines()
    papers = [[sent.strip() for sent in paper.split('##SENT##')] for paper in raw_papers]

    oracle_file = os.path.join(DATAPATH, DATA_TO_TEST[topic])
    with open(oracle_file, 'r') as stream:
        raw_labels = stream.readlines()

    golds = [literal_eval(raw_label) for raw_label in raw_labels]
    gold_sents = [[papers[i][index] for index in gold_label] for i, gold_label in enumerate(golds)]

    for i, summary in enumerate(summaries):
        predict_sents.append(summary.summary)
        predicts.append([papers[i].index(sent) for sent in summary.summary if sent in papers[i]])
        # predicts.append([summary.texts.index(sent) for sent in summary.summary])
    
    # score_p_r_f1 = calculate_performance(predicts, golds)
    score_p_r_f1 = 'unable to calculate now'
    # score_bleu = compute_bleu_raw(gold_sents, predict_sents, golds, predicts)
    score_bleu = compute_bleu_raw(gold_sents, predict_sents)

    return score_p_r_f1, score_bleu


#count_score
#score_func = ones#get_w2v_score_func()#get_tfidf_score_func()#count_score

# summaries = [Summary(texts, abstracts, query) for texts, abstracts, query in duck_iterator(duc_num)]
if __name__ == "__main__":

    if VERBOSE:
        call(['mkdir', '-p', 'test/temp_file'])
        call(['mkdir', '-p', os.path.join(OUTPUT_PATH, TARGET)])
    else:
        call(['mkdir', '-p', 'test/temp_file'], stdout=CallLog, stderr=STDOUT)
        call(['mkdir', '-p', os.path.join(OUTPUT_PATH, TARGET)], stdout=CallLog, stderr=STDOUT)



    for topic in DATA_TO_TEST.keys():
        for score_func_to_use in ['RSA-word2vec', 'RSA-TFIDF']:
            output_dir = os.path.join(OUTPUT_PATH, TARGET, score_func_to_use, topic)
            parse = True
            if os.path.exists(output_dir):
                if VERBOSE:
                    override = raw_input('{} of {} result exist, override (delete and inference again)? (Y/n): '.format(topic.capitalize(), score_func_to_use))
                else:
                    override = 'y'

                if override.lower() == 'y':
                    shutil.rmtree(output_dir)
                else:
                    parse = False


            if score_func_to_use == 'RSA-word2vec':
                score_func = get_w2v_score_func()
            elif score_func_to_use == 'RSA-TFIDF':
                score_func = get_tfidf_score_func(topic)
            if VERBOSE:
                print 'Getting summary objects...'
            summaries = [Summary(texts, None, query, score_func) for texts, query in paper_iterator(topic)]

            if parse:
                max_word_num = 0

                if VERBOSE:
                    print 'Calculating score and write it as tensorflow object...'
                with open('test/temp_file/%s' % topic, 'wb') as writer:
                    for summ in summaries:
                        article, abstract, scores = summ.get()
                        write_to_file(article, abstract, scores, writer)
                        max_word_num = max(max_word_num, summ.word_num)

                if VERBOSE:
                    print 'Maximum word number of this topic is', max_word_num
                    call(['rm', '-r', generated_path])
                    print 'Getting summarization...'
                    print 'Running command', cmd.format(topic, max_enc)
                    call(cmd.format(topic, max_enc).split())
                else:
                    call(['rm', '-r', generated_path], stdout=CallLog, stderr=STDOUT)
                    call(cmd.format(topic, max_enc).split(), stdout=CallLog, stderr=STDOUT)
                
                if VERBOSE:
                    print 'Summarization has generated to', generated_path, 'and copied to', output_dir
                    call(['mkdir', '-p', os.path.join(OUTPUT_PATH, TARGET, score_func_to_use)])
                    call(['cp', '-r', generated_path, output_dir])
                else:
                    call(['mkdir', '-p', os.path.join(OUTPUT_PATH, TARGET, score_func_to_use)], stdout=CallLog, stderr=STDOUT)
                    call(['cp', '-r', generated_path, output_dir], stdout=CallLog, stderr=STDOUT)
            
            generated_summaries = get_summaries(output_dir)

            for i in xrange(len(summaries)):
                summaries[i].add_sum(generated_summaries[i])

            if VERBOSE:
                print 'Evaluating', topic, 'of', score_func_to_use, '...'
            score_p_r_f1, score_bleu = evaluate(summaries, topic)
            # print 'Performance of', topic, 'of', score_func_to_use, 'is', score_p_r_f1, score_bleu
            print 'Performance of', topic, 'of', score_func_to_use, 'is', score_bleu

    if not VERBOSE:
        CallLog.close()
