import os
import shutil
from tqdm import tqdm
from path import Path
from lexrank import STOPWORDS, LexRank
from typing import List, Tuple, Dict, Iterable, Union
from ast import literal_eval as make_tuple
import nltk
import math

# RAW_DATAPATH = '../../../AnnotationAlignment/data/annotate/test'
DATAPATH = '../../../data/test'
DATA_PATH = '../data'
OUTPUT_PATH = '../output'
TARGET = 'LexRank'

DATA_DIR = os.path.join(DATA_PATH, TARGET)
OUTPUT_DIR = os.path.join(OUTPUT_PATH, TARGET)

THRESHOLD = 0.1
# SUMMARY_SIZE = 5
# PERCENT = None
PERCENT = 0.15


DATA_TO_TEST = {
    'Future': 'future/test.txt.oracle',
    'Contribution': 'contribution/test.txt.oracle',
    'Baseline': 'baseline/test.txt.oracle',
    'Dataset': 'dataset/test.txt.oracle',
}

TEST_PAPER = {key: value.replace('oracle', 'src')
              for key, value in DATA_TO_TEST.items()}
TEST_PAPER_SECTION = {key: value.replace(
    'oracle', 'section') for key, value in DATA_TO_TEST.items()}
PAPER_REF = {key: value.replace('oracle', 'ref')
             for key, value in DATA_TO_TEST.items()}

# Copied from Config.py
# TEST_IDS = [1, 10, 102, 103, 104, 105, 106, 107, 108, 11, 110, 111, 112, 113, 115, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42,
#             43, 44, 45, 46, 47, 49, 5, 50, 51, 52, 53, 55, 56, 57, 58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 7, 71, 72, 73, 74, 75, 76, 77, 78, 79, 8, 80, 82, 83, 84, 85, 86, 88, 89, 9, 92, 94, 95, 96, 97, 98]

# Copied from Paper2PPT/neusum_pt/loglinear/Config.py
PossibleSection = {
    "baseline": ["experiment", "experiments", "introduction", "baseline", "results", "result", "model", "related work"],
    "dataset": ["dataset", "experiment", "experiments"],
    "future": ["conclusion", "future work"],
    "contribution": ["abstract", "introduction"]
}

# https://stackoverflow.com/questions/33945261/how-to-specify-multiple-return-types-using-type-hints/33945518
# https://stackoverflow.com/questions/40181344/how-to-annotate-types-of-multiple-return-values


def parse_data(topic: str = 'Overall') -> Iterable[Union[List[List[str]], List[List[str]]]]:
    """ parse our annotation and write text files into DATA_DIR """

    with open(os.path.join(DATAPATH, TEST_PAPER[topic]), 'r') as stream:
        raw_papers = stream.readlines()
    papers = [paper.strip().split('##SENT##') for paper in raw_papers]
    with open(os.path.join(DATAPATH, TEST_PAPER_SECTION[topic]), 'r') as stream:
        raw_paper_sections = stream.readlines()
    sections = [paper.strip().split('##SENT##')
                for paper in raw_paper_sections]

    assert len(papers) == len(sections)

    return papers, sections


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


# Tuple(List[List[str]], List[Dict[int, int]]):
def get_topic_sentences(topic: str, papers: List[List[str]], sections: List[List[str]]) -> List[List[str]]:
    sections_to_select = PossibleSection.get(topic.lower(), None)
    if not sections_to_select:
        # , [{i: i  for i in range(len(paper))} for paper in papers]
        return papers

    new_papers = []
    for paper, section in zip(papers, sections):
        new_papers.append([sent for sent, sec in zip(
            paper, section) if sec.lower() in sections_to_select])
        # if not new_papers[-1]:
        #     import ipdb; ipdb.set_trace()

    return new_papers


def get_lxr(papers):
    """ Load parsed data and get lxr object """
    print('Getting LexRank object...')

    # We don't generate *.txt file now
    # documents = []
    # documentd_dir = Path(DATA_DIR)
    # for file_path in tqdm(documentd_dir.files('*.txt')):
    #     with file_path.open(mode='rt', encoding='utf-8') as fp:
    #         documents.append(fp.readlines())

    documents = ['\n'.join(paper) for paper in papers]
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])

    return lxr


def calculate_lxr(papers: List[List[str]], lxr: LexRank, percent: float = PERCENT):
    print('Calculating LexRank summary...')
    summaries = []
    sent_amount = []
    for paper in tqdm(papers):
        if not paper:
            summary = []
            sent_amount.append(0)
        else:
            if percent is None or percent <= 0:
                summary = lxr.get_summary(
                    paper, summary_size=min(len(paper), SUMMARY_SIZE), threshold=THRESHOLD)
                sent_amount.append(min(len(paper), SUMMARY_SIZE))
            else:
                summary = lxr.get_summary(
                    paper, summary_size=math.ceil(percent * len(paper)), threshold=THRESHOLD)
                sent_amount.append(math.ceil(percent * len(paper)))

        summaries.append(summary)
    return summaries, sent_amount


def summaries_to_sent_ids(papers: List[List[str]], summaries: List[List[str]]):
    print('Converting summaries to ids...')
    sent_ids = []
    for paper, summary in tqdm(zip(papers, summaries), total=len(papers)):
        temp_sent = []
        for sentence in summary:
            temp_sent.append(paper.index(sentence))
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


def eval_write_output(topic: str, summaries: List[List[str]], sent_ids: List[List[int]], topic_id_to_id: Dict[str, List[int]], sent_amount: List[int], topic_papers: List[List[str]], lxr: LexRank):
    """ evaluate and write sentence prediction and score """
    print('Writting output...')
    # if os.path.exists(OUTPUT_DIR):
    #     shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Predict
    if PERCENT is None or PERCENT <= 0:
        predicts_file = os.path.join(
            OUTPUT_DIR, f'{topic}_prediction_{SUMMARY_SIZE}_{THRESHOLD}.txt')
    else:
        predicts_file = os.path.join(
            OUTPUT_DIR, f'{topic}_prediction_{PERCENT}_{THRESHOLD}.txt')

    predict_fp = open(predicts_file, 'w')

    for summary, indices in tqdm(zip(summaries, sent_ids), total=len(summaries)):
        scores_cont = lxr.rank_sentences(
            summary,
            threshold=THRESHOLD
        )

        predict_fp.write(f'{summary} {indices} {scores_cont}\n')

    predict_fp.close()

    # Performance

    if PERCENT is None or PERCENT <= 0:
        performance_file = os.path.join(
            OUTPUT_DIR, f'{topic}_performance_{SUMMARY_SIZE}_{THRESHOLD}.txt')
    else:
        performance_file = os.path.join(
            OUTPUT_DIR, f'{topic}_performance_{PERCENT}_{THRESHOLD}.txt')

    performance_fp = open(performance_file, 'w')

    if topic == 'Overall':
        # test on each topic with entire paper set
        for topic, test_file in DATA_TO_TEST.items():
            with open(os.path.join(DATAPATH, test_file), 'r') as stream:
                raw_labels = stream.readlines()
            labels = [make_tuple(raw_label) for raw_label in raw_labels]

            # mapping to match topic papers
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
    else:
        with open(os.path.join(DATAPATH, DATA_TO_TEST[topic]), 'r') as stream:
            raw_labels = stream.readlines()
        labels = [make_tuple(raw_label) for raw_label in raw_labels]

        # pred_ids = [sent_ids[topic_id_to_id[topic][i]] for i in range(len(raw_labels))]
        pred_ids = sent_ids
        performance = calculate_performance(pred_ids, labels)

        gold_sents = [[topic_papers[i][index] for index in gold_label]
                      for i, gold_label in enumerate(labels)]
        predict_sents = [[topic_papers[i][index] for index in predict_label]
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
    # might have potential bug of different topic
    for topic in DATA_TO_TEST.keys():
        papers, sections = parse_data(topic)
        topic_id_to_id = get_id_match()
        new_papers = get_topic_sentences(topic, papers, sections)
        print('=== Empty paper:', sum(paper == []
                                      for paper in new_papers), '; total in topic:', len(new_papers), topic)
        lxr = get_lxr(new_papers)
        summaries, sent_amount = calculate_lxr(new_papers, lxr)
        sent_ids = summaries_to_sent_ids(papers, summaries)
        eval_write_output(topic, summaries, sent_ids,
                          topic_id_to_id, sent_amount, papers, lxr)
