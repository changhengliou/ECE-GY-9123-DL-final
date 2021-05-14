import os
from collections import defaultdict
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
from ast import literal_eval as make_tuple
import nltk

DATAPATH = '../../../data/test'
TARGET = 'MEAD_TEST'
OUTPUT_PATH = '../output'
OUTPUT_DIR = os.path.join(OUTPUT_PATH, 'mead')
# PERCENT = 3
PERCENT = 15


DATA_TO_TEST = {
    'Future': 'future/test.txt.oracle',
    'Contribution': 'contribution/test.txt.oracle',
    'Baseline': 'baseline/test.txt.oracle',
    'Dataset': 'dataset/test.txt.oracle',
}

TEST_PAPER = {key: value.replace('oracle', 'src')
              for key, value in DATA_TO_TEST.items()}
PAPER_REF = {key: value.replace('oracle', 'ref')
             for key, value in DATA_TO_TEST.items()}


def parse_prediction():
    extract_file = os.path.join(OUTPUT_DIR, f'{TARGET}.extract')
    tree = ET.parse(extract_file)
    root = tree.getroot()

    prediction = defaultdict(list)
    for item in root.getchildren():
        # SNO index from 1
        # sentence id index from 0
        prediction[int(item.get('DID'))].append(int(item.get('SNO')) - 1)

    predicts = []
    # assume the testset indices are 1 ~ 100
    for i in range(100):
        predicts.append(prediction[i + 1])

    return predicts


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

# COPY from LexRank


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

        gs = [sent for sent, _ in sorted(
            zip(gs, gid), key=lambda pair: pair[1])]
        ps = [sent for sent, _ in sorted(
            zip(ps, pid), key=lambda pair: pair[1])]

        # TODO: .split() can be replace with any other tokenizer
        # TODO: whether concat sentences
        references = [sent.split() for sent in gs]
        hypothesis = [word for sent in ps for word in sent.split()]

        # for avg_sent_bleu
        try:
            avg_sent_bleu += nltk.translate.bleu_score.sentence_bleu(references, hypothesis,
                                                                     smoothing_function=chencherry.method1)
        except:
            import ipdb
            ipdb.set_trace()

        # for corpus_bleu
        all_references.append(references)
        all_hypothesis.append(hypothesis)

    # corpus_bleu() is different from averaging sentence_bleu() for hypotheses
    # bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis,
    #                                              smoothing_function=chencherry.method1)
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(
        all_references, all_hypothesis)
    avg_sent_bleu /= len(gold_sents)

    return {'corpus_bleu': corpus_bleu, 'avg_sent_bleu': avg_sent_bleu}


######


def eval_write_output(sent_ids: List[List[int]], topic_id_to_id: Dict[str, List[int]], sent_amount: List[int]):
    """ evaluate and write evaluation score """
    print('Writting output...')

    performance_file = os.path.join(
        OUTPUT_DIR, f'performance_{PERCENT}_percent.txt')

    performance_fp = open(performance_file, 'w')

    for topic, test_file in DATA_TO_TEST.items():
        with open(os.path.join(DATAPATH, test_file), 'r') as stream:
            raw_labels = stream.readlines()
        labels = [make_tuple(raw_label) for raw_label in raw_labels]

        pred_ids = [sent_ids[topic_id_to_id[topic][i]]
                    for i in range(len(raw_labels))]
        performance = calculate_performance(pred_ids, labels)

        with open(os.path.join(DATAPATH, TEST_PAPER[topic]), 'r') as stream:
            raw_papers = stream.readlines()
        papers = [paper.strip().split('##SENT##') for paper in raw_papers]

        gold_sents = [[papers[i][index] for index in gold_label]
                      for i, gold_label in enumerate(labels)]
        predict_sents = [[papers[i][index] for index in predict_label]
                         for i, predict_label in enumerate(pred_ids)]

        bleu_performance = compute_bleu_raw(
            gold_sents, predict_sents, labels, pred_ids)

        print(topic, performance, bleu_performance)
        performance_fp.write(f'{topic} {performance} {bleu_performance}\n')

    total_sents = sum(sent_amount)
    print(f'Total sentences: {total_sents}')
    performance_fp.write(f'Total sentences: {total_sents}\n')
    print(f'Total average per paper: {total_sents/len(sent_amount)}')
    performance_fp.write(
        f'Total average per paper: {total_sents/len(sent_amount)}\n')
    print(f'Sentences for each paper: {sent_amount}')
    performance_fp.write(f'Sentences for each paper: {sent_amount}\n')

    performance_fp.close()


if __name__ == "__main__":
    sent_ids = parse_prediction()
    topic_id_to_id = get_id_match()
    sent_amount = [len(sent) for sent in sent_ids]
    eval_write_output(sent_ids, topic_id_to_id, sent_amount)
