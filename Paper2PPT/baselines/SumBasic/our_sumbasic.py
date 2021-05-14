import nltk
import sys
import glob
import os
import shutil
from tqdm import tqdm
from parse_data import DATA_DIR, OUTPUT_DIR, DATAPATH
from typing import List, Tuple, Dict
from ast import literal_eval as make_tuple
import nltk

method = sys.argv[1]

lemmatize = True
rm_stopwords = True
# NUM_SENTENCES = 5
# PERCENT = None
PERCENT = 0.15
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


DATA_TO_TEST = {
    'Overall': 'test.txt.oracle',
    'Future': 'future/test.txt.oracle',
    'Contribution': 'contribution/test.txt.oracle',
    'Baseline': 'baseline/test.txt.oracle',
    'Dataset': 'dataset/test.txt.oracle',
    'Metric': 'metric/test.txt.oracle',
    'Motivation': 'motivation/test.txt.oracle'
}

TEST_PAPER = {key: value.replace('oracle', 'src') for key, value in DATA_TO_TEST.items()}
PAPER_REF = {key: value.replace('oracle', 'ref') for key, value in DATA_TO_TEST.items()}

def get_id_match() -> Dict[str, List[int]]:
    """ some topic might not exist in the entire test set,
    as we only store the sample which has at least one gold,
    some sample might be skipped. so we have to match them """

    with open(os.path.join(DATAPATH, PAPER_REF['Overall']), 'r') as stream:
        test_ref = stream.readlines()
    ref_to_id = {ref.strip(): index for index, ref in enumerate(test_ref)}

    topic_id_to_id = {}

    for topic, ref_file in PAPER_REF.items():
        with open(os.path.join(DATAPATH, ref_file), 'r') as stream:
            topic_ref = stream.readlines()
        
        topic_id_to_id[topic] = [ref_to_id[ref.strip()] for ref in topic_ref]
    
    return topic_id_to_id
    
def clean_sentence(tokens):
    # print('Cleaning sentences...')
    tokens = [t.lower() for t in tokens]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if rm_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def get_probabilities(cluster, lemmatize, rm_stopwords):
    # print('Getting probabilities...')
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
    # print('Getting sentences...')
    sentences = []
    for path in cluster:
        with open(path) as f:
            sentences += [line.strip() for line in f.readlines()]
    return sentences


def clean_sentence(tokens):
    tokens = [t.lower() for t in tokens]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    if rm_stopwords:
        tokens = [t for t in tokens if t not in stopwords]
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
        if max_score == None or score > max_score:
            max_sentence = sentence
            max_score = score
    if not simplified:
        update_ps(max_sentence, word_ps)
    return max_sentence, score


def update_ps(max_sentence, word_ps):
    sentence = nltk.word_tokenize(max_sentence)
    sentence = clean_sentence(sentence)
    for word in sentence:
        try:
            word_ps[word] = word_ps[word]**2
        except:
            print(f'warning: word "{word}" not found in word_ps')
    return True


def orig(cluster):
    cluster = glob.glob(cluster)
    word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
    sentences = get_sentences(cluster)
    summary = []

    if PERCENT is not None and PERCENT > 0:
        num_sentences = int(len(sentences) * PERCENT)
    else:
        num_sentences = NUM_SENTENCES

    for i in range(num_sentences):
        summary.append(max_sentence(sentences, word_ps, False))
    return summary, sentences, num_sentences


def simplified(cluster):
    cluster = glob.glob(cluster)
    word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
    sentences = get_sentences(cluster)
    summary = []

    if PERCENT is not None and PERCENT > 0:
        num_sentences = int(len(sentences) * PERCENT)
    else:
        num_sentences = NUM_SENTENCES

    for i in range(num_sentences):
        summary.append(max_sentence(sentences, word_ps, True))
    return summary, sentences, num_sentences


def leading(cluster):
    cluster = glob.glob(cluster)
    sentences = get_sentences(cluster)
    summary = []

    if PERCENT is not None and PERCENT > 0:
        num_sentences = int(len(sentences) * PERCENT)
    else:
        num_sentences = NUM_SENTENCES

    for i in range(num_sentences):
        summary.append(sentences[i])
    return summary, sentences, num_sentences

# COPY from LexRank


def summaries_to_sent_ids(papers: List[List[str]], summaries: List[List[str]]):
    print('Converting summaries to ids...')
    sent_ids = []
    for paper, summary in tqdm(zip(papers, summaries), total=len(papers)):
        temp_sent = []
        for sentence in summary:
            try:
                temp_sent.append(paper.index(sentence))
            except:
                print(f'warning: can\'t find sentence "{sentence}"')
        sent_ids.append(temp_sent)
    return sent_ids


def calculate_performance(predicts: List[Tuple[int]], golds: List[Tuple[int]]):
    print('Calculating performance...')
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

def compute_bleu_raw(gold_sents: List[Tuple[str]], predict_sents: List[Tuple[str]], gold: List[Tuple[int]], predict: List[Tuple[int]]):
    """ input raw sentences (precision)
    https://www.nltk.org/api/nltk.translate.html
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://www.nltk.org/_modules/nltk/align/bleu_score.html
    https://stackoverflow.com/questions/32395880/calculate-bleu-score-in-python/39062009
    """
    # for avg_sent_bleu
    avg_sent_bleu = 0
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    # for corpus_bleu
    all_references = []
    all_hypothesis = []

    for gs, ps, gid, pid in zip(gold_sents, predict_sents, gold, predict):

        gs = [sent for sent, _ in sorted(zip(gs, gid), key=lambda pair: pair[1])]
        ps = [sent for sent, _ in sorted(zip(ps, pid), key=lambda pair: pair[1])]

        # TODO: .split() can be replace with any other tokenizer
        references = [sent.split() for sent in gs]  # TODO: whether concat sentences
        hypothesis = [word for sent in ps for word in sent.split()]

        # for avg_sent_bleu
        try:
            avg_sent_bleu += nltk.translate.bleu_score.sentence_bleu(references, hypothesis,
                                                            smoothing_function=chencherry.method1)
        except:
            import ipdb; ipdb.set_trace()
        
        # for corpus_bleu
        all_references.append(references)
        all_hypothesis.append(hypothesis)

    # corpus_bleu() is different from averaging sentence_bleu() for hypotheses
    # bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis,
    #                                              smoothing_function=chencherry.method1)
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis)
    avg_sent_bleu /= len(gold_sents)

    return {'corpus_bleu': corpus_bleu, 'avg_sent_bleu': avg_sent_bleu}



def eval_write_output(summaries: List[List[str]], sent_ids: List[List[int]], topic_id_to_id: Dict[str, List[int]], scores: List[List[int]], sent_amount: List[int]):
    """ evaluate and write sentence prediction and score """
    print('Writting output...')
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Predict
    if PERCENT is not None and PERCENT > 0:
        predicts_file = os.path.join(
            OUTPUT_DIR, f'prediction_{method}_{PERCENT}.txt')
    else:
        predicts_file = os.path.join(
            OUTPUT_DIR, f'prediction_{method}_{NUM_SENTENCES}.txt')

    predict_fp = open(predicts_file, 'w')

    for summary, scores_cont in tqdm(zip(summaries, scores), total=len(summaries)):
        predict_fp.write(f'{summary} {scores_cont}\n')

    predict_fp.close()

    # Performance

    if PERCENT is not None and PERCENT > 0:
        performance_file = os.path.join(
            OUTPUT_DIR, f'performance_{method}_{PERCENT}.txt')
    else:
        performance_file = os.path.join(
            OUTPUT_DIR, f'performance_{method}_{NUM_SENTENCES}.txt')
    
    performance_fp = open(performance_file, 'w')

    for topic, test_file in DATA_TO_TEST.items():
        with open(os.path.join(DATAPATH, test_file), 'r') as stream:
            raw_labels = stream.readlines()
        labels = [make_tuple(raw_label) for raw_label in raw_labels]

        pred_ids = [sent_ids[topic_id_to_id[topic][i]] for i in range(len(raw_labels))]
        performance = calculate_performance(pred_ids, labels)

        with open(os.path.join(DATAPATH, TEST_PAPER[topic]), 'r') as stream:
            raw_papers = stream.readlines()
        papers = [paper.strip().split('##SENT##') for paper in raw_papers]

        gold_sents = [[papers[i][index] for index in gold_label] for i, gold_label in enumerate(labels)]
        # TODO: somehow the prediction index exceed range
        predict_sents = [[papers[i][index] for index in predict_label if index < len(papers[i])] for i, predict_label in enumerate(pred_ids)]

        bleu_performance = compute_bleu_raw(gold_sents, predict_sents, labels, pred_ids)

        print(topic, performance, bleu_performance)
        performance_fp.write(f'{topic} {performance} {bleu_performance}\n')

    total_sents = sum(sent_amount)
    print(f'Total sentences: {total_sents}')
    performance_fp.write(f'Total sentences: {total_sents}\n')
    print(f'Total average per paper: {total_sents/len(sent_amount)}')
    performance_fp.write(f'Total average per paper: {total_sents/len(sent_amount)}\n')
    print(f'Sentences for each paper: {sent_amount}')
    performance_fp.write(f'Sentences for each paper: {sent_amount}\n')

    performance_fp.close()

#################


def main():
    # method = 'orig'
    cluster = os.path.join(DATA_DIR, '*.txt')
    papers = []
    scores = []
    summaries = []
    sent_amount = []
    for filename in tqdm(glob.glob(cluster)):
        result, sentences, num_sentences = eval(method + "('" + filename + "')")
        summary, score = list(zip(*result))
        papers.append(sentences)
        scores.append(score)
        summaries.append(summary)
        sent_amount.append(num_sentences)

    topic_id_to_id = get_id_match()
    sent_ids = summaries_to_sent_ids(papers, summaries)

    eval_write_output(summaries, sent_ids, topic_id_to_id, scores, sent_amount)


if __name__ == '__main__':
    main()
