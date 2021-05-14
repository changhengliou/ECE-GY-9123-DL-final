from __future__ import division

import os
from glob import glob
import math
import random
import time
import logging
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F
from ast import literal_eval as make_tuple
from typing import List, Tuple
from tensorboardX import SummaryWriter

import neusum
from neusum.xinit import xavier_normal, xavier_uniform
import loglinear
import xargs
import nltk
# This can't be installed by pip
# https://github.com/pcyin/PyRouge
# from PyRouge.Rouge import Rouge

parser = argparse.ArgumentParser(description='train.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

# If set the level to DEBUG will show lots of log of onlinePreprocessing
logging.basicConfig(
    format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
if len(logging.root.handlers) == 1:  # only default handler
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
    if opt.log_home:
        log_file_name = os.path.join(opt.log_home, log_file_name)

    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
    logging.root.addHandler(file_handler)

    logger.info('My PID is {0}'.format(os.getpid()))
    logger.info(opt)

    if torch.cuda.is_available() and not opt.gpus:
        logger.info(
            "WARNING: You have a CUDA device, so you should probably run with -gpus 0")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if opt.gpus:
        if opt.cuda_seed > 0:
            torch.cuda.manual_seed(opt.cuda_seed)
        cuda.set_device(opt.gpus[0])
        logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

    logger.info('My seed is {0}'.format(torch.initial_seed()))

writer = SummaryWriter(logdir=opt.save_path)

# prevent from additional logger so import here
from eval_loglinear import evalModel as evalLoglinearModel

# Some parameters protection
if opt.significant_criteria == 'random_ratio':
    assert opt.add_all_bad_data_after >= opt.use_good_data_only_before, \
        "-use_good_data_only_before (include) should be in front of -add_bad_data_after (exclude)"
else:
    logger.warning('Auto set -use_good_data_only_before to -epochs')
    logger.warning('Auto set -stop_em_and_loglinear_after to -epochs + 1')
    opt.use_good_data_only_before = opt.epochs
    opt.stop_em_and_loglinear_after = opt.epochs + 1


def NMTCriterion():
    # weight[s2s.Constants.PAD] = 0
    # crit = nn.NLLLoss(weight, size_average=False)
    # size_average & reduce arguments are deprecated
    # crit = nn.NLLLoss(size_average=False)
    # crit = nn.NLLLoss(reduction='sum')
    crit = nn.NLLLoss(reduction='none')
    if opt.gpus:
        crit.cuda()
    return crit


def compute_nll_loss(pred_scores, gold_scores, mask, crit, bad_indices=None):
    """

    :param pred_scores: (step, batch, doc_len)
    :param gold_scores: (batch*step, doc_len)
    :param mask: (batch, doc_len)
    :param crit:
    :return:
    """
    pred_scores = pred_scores.view(-1, pred_scores.size(2))

    # Because the pred_scores of the pointer model has already been softmaxed
    pred_scores = torch.log(pred_scores + 1e-8)
    # pred_scores = torch.log_softmax(pred_scores + 1e-8, dim=-1)

    gold_scores = gold_scores.view(-1)

    loss = crit(pred_scores, gold_scores)
    # works the same as NLLLoss
    # loss = -torch.gather(pred_scores, 1, gold_scores.view(-1, 1)).view(-1)

    if bad_indices is not None:
        to_keep = torch.tensor(
            [i not in bad_indices for i in range(len(loss))]).bool()
        if opt.gpus:
            to_keep = to_keep.cuda()

        loss = loss.masked_select(to_keep)

    return loss.sum()


def regression_loss(pred_scores, gold_scores, mask, crit, teach_student: bool = True):
    """
    :param pred_scores: (step, batch, doc_len) => pointer network output has already been softmaxed
    :param gold_scores: (batch*step, doc_len) => non-softmaxed logits
    :param mask: (batch, doc_len)
    :param crit: torch loss function
    :param teach_student: if true, then use gold_scores (log-linear model) as gold other wise reverse it
    (When use log-linear model, use EM to update teacher and student rotary)
    :return:
    """
    # pred_scores = pred_scores.transpose(
    #     0, 1).contiguous()  # (batch, step, doc_len)
    pred_scores = pred_scores.view(-1, pred_scores.size(2))
    # gold_scores = gold_scores.view(*pred_scores.size())
    assert isinstance(crit, nn.KLDivLoss)

    # similar to modules.ScoreAttention.py
    gold_scores = gold_scores * (1 - mask) + mask * (-1e8)

    if teach_student:
        # Because the pred_scores of the pointer model has already been softmaxed
        pred_scores = torch.log(pred_scores + 1e-8)
        # pred_scores = torch.log_softmax(pred_scores + 1e-8, dim=-1)

        gold_scores = torch.softmax(gold_scores, dim=-1)

    else:
        # student teach teacher
        pred_scores, gold_scores = torch.log_softmax(
            gold_scores, dim=-1), pred_scores

    # Disable gold_scores' gradient, or it will somehow confuse the loss.backward()
    gold_scores = Variable(gold_scores, requires_grad=False)
    # for KLDivLoss, the gold_score don't need to apply log
    # https://pytorch.org/docs/stable/nn.html#kldivloss
    # https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/6
    loss = crit(pred_scores, gold_scores)

    # RuntimeError: expand(torch.cuda.FloatTensor{[32, 1, 500]}, size=[32, 500]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
    # loss = loss * (1 - mask).unsqueeze(1).expand_as(loss)
    loss = loss * (1 - mask).view_as(loss)
    reduce_loss = loss.sum()  # reduce=False, sum manually
    return reduce_loss


def load_dev_data(summarizer, src_file: str, oracl_file: str, src_section_file: str, drop_too_short: int = 10, drop_too_long: int = 500, test_bert_annotation: str = '', postfix: str = '', qtype: str = ''):
    """
    Load dev/test set data. (similar with makeData in onlinePreprocess.py)
    """

    def addPair(f1, f2, f3, f4=None):
        if not f4:
            for x, x2, y1 in zip(f1, f2, f3):
                yield (x, x2, y1, None)
            yield (None, None, None, None)
        else:
            for x, x2, y1, y2 in zip(f1, f2, f3, f4):
                yield (x, x2, y1, y2)
            yield (None, None, None, None)

    if postfix:
        # assert if postfix is given, then it is running on train data
        keywords = loglinear.Config.Keyword[qtype]
        use_good = True
    else:
        # normal case
        keywords = []
        use_good = False

    # here tgt is sentence index
    seq_length = opt.max_sent_length
    dataset, raw = [], []
    src_raw, tgt_raw = [], []
    src_section_raw, src_section_batch = [], []
    src_batch, tgt_batch = [], []
    oracle_batch = []
    srcF = open(src_file, encoding='utf-8')
    srcSectionF = open(src_section_file, encoding='utf-8')
    # tgtF = open(tgt_file, encoding='utf-8')
    oracleF = open(oracl_file, encoding='utf-8')

    if test_bert_annotation:
        bertF = open(test_bert_annotation, encoding='utf-8')
        bert_annotation_batch = []
    else:
        bertF = None

    for sline, secline, oline, bline in addPair(srcF, srcSectionF, oracleF, bertF):
        # if (sline is not None) and (tline is not None):
        if (sline is not None) and (oline is not None):
            if sline == "" or oline == "":
                continue
            sline = sline.strip()
            secline = secline.strip()
            oline = oline.strip()
            if test_bert_annotation:
                bline = bline.strip()
            srcSents = sline.split('##SENT##')
            srcSectionSents = secline.split('##SENT##')

            if len(srcSents) < drop_too_short or len(srcSents) > drop_too_long:
                logger.info('Drop data too short or too long')
                continue

            # this will transfer string of tuple to tuple
            oracle_combination = make_tuple(oline.split('\t')[0])
            oracle_combination = [x for x in oracle_combination]  # no sentinel
            if test_bert_annotation:
                bert_annotation_combination = make_tuple(bline.split('\t')[0])
                bert_annotation_combination = [
                    x for x in bert_annotation_combination]  # no sentinel
            srcWords = [x.split(' ')[:seq_length] for x in srcSents]
            srcSectionWords = [x.split(' ')[:seq_length]
                               for x in srcSectionSents]
            # tgtWords = ' '.join(tgtSents)
            src_raw.append(srcSents)
            src_batch.append(srcWords)
            src_section_raw.append(srcSectionSents)
            src_section_batch.append(srcSectionWords)
            # tgt_raw.append(tgtWords)
            oracle_batch.append(torch.LongTensor(oracle_combination))
            if test_bert_annotation:
                bert_annotation_batch.append(
                    torch.LongTensor(bert_annotation_combination))

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        # data = summarizer.buildData(src_batch, src_raw, tgt_raw, None, None)
        if test_bert_annotation:
            data = summarizer.buildData(
                src_batch, src_raw, None, oracle_batch, None, src_section_batch, src_section_raw, bert_annotation=bert_annotation_batch, good_patterns=keywords, use_good=use_good)
        else:
            data = summarizer.buildData(
                src_batch, src_raw, None, oracle_batch, None, src_section_batch, src_section_raw, good_patterns=keywords, use_good=use_good)
        dataset.append(data)
        src_batch, tgt_batch = [], []
        src_raw, tgt_raw = [], []
        src_section_raw, src_section_batch = [], []
        oracle_batch = []
        if test_bert_annotation:
            bert_annotation_batch = []

    srcF.close()
    # tgtF.close()
    oracleF.close()
    if test_bert_annotation:
        bertF.close()

    return dataset


evalModelCount = 0
totalBatchCount = 0


def compute_selection_acc(gold: List[Tuple[int]], predict: List[Tuple[int]], hit1mode: bool = False):
    """ Compute sentence accuracy """
    if len(gold) != len(predict):
        logger.warning("Eval size mismatch: gold {0}, size {1}".format(
            len(gold), len(predict)))
        return 0
    corr, total = 0, 0
    for i in range(len(gold)):
        for j in range(len(gold[i])):
            if gold[i][j] in predict[i]:
                corr += 1
                if hit1mode:
                    break
        if hit1mode:
            total += 1
        else:
            total += len(gold[i])
    return corr / total


def compute_metric(gold: List[Tuple[int]], predict: List[Tuple[int]]):
    """ Compute precision, recall, f1 score """
    if len(gold) != len(predict):
        logger.warning("Eval size mismatch: gold {0}, size {1}".format(
            len(gold), len(predict)))
        return 0
    true_positive, total_predicted_positive, total_gold_positive = 0, 0, 0
    for i in range(len(gold)):
        for j in range(len(gold[i])):
            if gold[i][j] in predict[i]:
                true_positive += 1
        total_predicted_positive += len(predict[i])
        total_gold_positive += len(gold[i])
    precision = true_positive / total_predicted_positive
    recall = true_positive / total_gold_positive
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        'p': precision,
        'r': recall,
        'f1': f1
    }


# def compute_rouge(gold: List[Tuple[int]], predict: List[Tuple[int]]):
#     """ follow the old (original code) """
#     rouge_calculator = Rouge.Rouge(use_ngram_buf=True)
#     p, r, f1 = rouge_calculator.compute_rouge(gold, predict)
#     return {'rouge_p': p, 'rouge_r': r, 'rouge_f1': f1}
#
#
# def compute_rouge_raw(gold_sents: List[Tuple[str]], predict_sents: List[Tuple[str]]):
#     """ input raw sentences (recall) """
#     r = Rouge()
#
#     p, r, f1 = 0, 0, 0
#     for gs, ps in zip(gold_sents, predict_sents):
#         system_generated_summary = [sent for sent, section in gs]
#         manual_summary = [sent for sent, section in ps]
#         p_temp, r_temp, f1_temp = r.rouge_l(system_generated_summary, manual_summmary)
#         p += p_temp
#         r += r_temp
#         f1 += f1_temp
#
#     p /= len(gold_sents)
#     r /= len(gold_sents)
#     f1 /= len(gold_sents)
#
#     return {'rouge_raw_p': p, 'rouge_raw_r': r, 'rouge_raw_f1': f1}


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

        gs = [sent for (sent, _), _ in sorted(
            zip(gs, gid), key=lambda pair: pair[1])]
        ps = [sent for (sent, _), _ in sorted(
            zip(ps, pid), key=lambda pair: pair[1])]

        # TODO: .split() can be replace with any other tokenizer
        references = [sent.split() for sent in gs]  # TODO: whether concat sentences
        hypothesis = [word for sent in ps for word in sent.split()]

        # for avg_sent_bleu
        avg_sent_bleu += nltk.translate.bleu_score.sentence_bleu(references, hypothesis,
                                                        smoothing_function=chencherry.method1)
        
        # for corpus_bleu
        all_references.append(references)
        all_hypothesis.append(hypothesis)

    # corpus_bleu() is different from averaging sentence_bleu() for hypotheses
    # bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis,
    #                                              smoothing_function=chencherry.method1)
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(all_references, all_hypothesis)
    avg_sent_bleu /= len(gold_sents)

    return {'corpus_bleu': corpus_bleu, 'avg_sent_bleu': avg_sent_bleu}


def getLabel(summarizer, dataset, output_len: int = 1, threshold: float = 0.0001, stripping_mode: str = 'magnitude', isEval: bool = False, log_linear_model: nn.Module = None):
    """ get label for evalModel, evalTrainData and RelabelTrainData
        predict, gold, predict_sents won't be influenced by output_len

        TODO: involve log-linear model in decoding?!
    """
    # none: nothing will be stripped (label will always be as same length as max decode step)
    # normal: stripped all the label if probability less than the threshold
    # magnitude: is the difference of two probability is greater than the magnitude then strip it
    assert stripping_mode in ('none', 'normal', 'diff', 'magnitude')

    predict, gold, predict_sents, gold_sents = [], [], [], []
    attnScore, topkPred = [], []
    n_best = 1

    for i in range(len(dataset)):
        """
        input: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (oracleBatch, oracleLength), \
               Variable(torch.LongTensor(list(indices)).view(-1).cuda(), volatile=self.volatile)
        """
        batch = dataset[i]
        if isEval:
            batch = batch[0]

        if opt.max_decode_step > output_len:
            topk = opt.max_decode_step
        else:
            topk = output_len

        #  (2) translate
        # batch contain
        # 0: three tensors => (wrap(srcBatch), lengths, doc_lengths)
        # 1: all sentences in a batch (src_raw is a 2D matrix)
        # 2: two tensors
        # 3: a tensor
        pred, predScore, attn = summarizer.translateBatch(batch)
        # pred, predScore, attn = list(zip(
        #     *sorted(zip(pred, predScore, attn, indices),
        #             key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        oracleBatch = []

        src_raw = batch[1][0]
        src_section_raw = batch[7][0]
        predict_id, oracle_id = [], []
        gold_ids, gold_lengths = batch[2][0].tolist(
        ), batch[2][1].view(-1).tolist()

        for b in range(len(src_raw)):
            # In normal case, n is just 0
            for n in range(n_best):
                selected_sents = []
                selected_id = []
                # selected_id_score = []

                if stripping_mode == 'none':
                    for step, idx in enumerate(pred[b][n]):
                        # no sentinel
                        if idx >= len(src_raw[b]):
                            break
                        selected_sents.append(
                            (src_raw[b][idx], src_section_raw[b][idx]))
                        selected_id.append(idx.item())
                        # TODO: attempt to get the attntion score of selected_id (it is not always in descending order)
                        # selected_id_score.append(attn[b][n][step, idx.item()])

                if output_len > 0 or stripping_mode != 'none':
                    selected_topk_id = []
                    selected_attn_score = []
                    for attn_score in attn[b][n]:
                        values, indices = attn_score.topk(
                            k=topk, dim=0, largest=True, sorted=True)
                        selected_attn_score.append(values)
                        selected_topk_id.append(indices)

                # Currently, normal and magnitude mode use the attention score of "greedy" instead of beam search
                if stripping_mode == 'normal': # threshold
                    for j, score in enumerate(selected_attn_score[0]):
                        if score < threshold:
                            break

                    # make sure at least one output
                    if j < 1:
                        j = 1

                    j = min(j, opt.max_decode_step)

                    for idx in selected_topk_id[0][:j]:
                        selected_sents.append(
                            (src_raw[b][idx], src_section_raw[b][idx]))
                        selected_id.append(idx.item())

                elif stripping_mode == 'diff': # threshold of difference
                    for j in range(1, len(selected_attn_score[0])):
                        if selected_attn_score[0][j - 1] - selected_attn_score[0][j] > threshold:
                            break

                    j = min(j, opt.max_decode_step)

                    for idx in selected_topk_id[0][:j]:
                        selected_sents.append(
                            (src_raw[b][idx], src_section_raw[b][idx]))
                        selected_id.append(idx.item())

                elif stripping_mode == 'magnitude':
                    # make sure at least one output
                    j = 1

                    for j in range(1, len(selected_attn_score[0])):
                        if (torch.log10(selected_attn_score[0][j - 1]) - torch.log10(selected_attn_score[0][j])) > -torch.log10(torch.tensor(threshold)):
                            break

                    j = min(j, opt.max_decode_step)

                    for idx in selected_topk_id[0][:j]:
                        selected_sents.append(
                            (src_raw[b][idx], src_section_raw[b][idx]))
                        selected_id.append(idx.item())

                # the original code (preserve this for accuracy calculation)
                if n == 0:
                    predict_id.append(tuple(selected_id))
                    predBatch.append(tuple(selected_sents))

                    oracle_id.append(tuple(gold_ids[b][:gold_lengths[b]]))
                    # TODO: gold idx shouldn't exceed the raw text...
                    # oracleBatch.append(
                    #     tuple((src_raw[b][idx], src_section_raw[b][idx]) for idx in oracle_id[-1]))
                    oracleBatch.append(
                        tuple((src_raw[b][idx], src_section_raw[b][idx]) for idx in oracle_id[-1] if idx < len(src_raw[b])))

                    if output_len > 0:
                        attnScore.append(
                            tuple(selected_attn_score[:output_len]))
                        topkPred.append(tuple(selected_topk_id[:output_len]))

        # tgt_raw = batch[2][0]
        # gold += tgt_raw
        gold_sents += oracleBatch
        gold += oracle_id
        predict_sents += predBatch
        predict += predict_id

    return predict, gold, predict_sents, attnScore, topkPred, gold_sents


def RelabelTrainData(summarizer, trainData, output_len: int):
    """ for iterative labeling
    TODO: this might cause bug for log-linear model
    """
    predict, _, _, _, _, _ = getLabel(
        summarizer, trainData, output_len, threshold=opt.threshold, stripping_mode=opt.stripping_mode)

    translabel = [torch.LongTensor(x) for x in predict]
    logger.info('Relabeled')
    trainData.oracle = translabel


def evalTrainData(model, summarizer, trainData, epoch_num: int = -1, output_len: int = 1):
    """ evaluate training data and output train.out.i """
    # model didn't be used
    global evalModelCount

    predict, gold, predict_sents, attnScore, topkPred, gold_sents = getLabel(
        summarizer, trainData, output_len, threshold=opt.threshold, stripping_mode=opt.stripping_mode)

    # scores = compute_rouge(gold, predict)
    scores = compute_selection_acc(gold, predict)

    if epoch_num >= 0:
        index = epoch_num
    else:
        index = evalModelCount

    with open(os.path.join(opt.save_path, 'train.out.{0}'.format(index)), 'w', encoding='utf-8') as of:
        for p, sent in zip(predict, predict_sents):
            of.write('{0}\t{1}'.format(sent, p) + '\n')

    if output_len > 1:
        with open(os.path.join(opt.save_path, 'train.{1}_n_out.{0}'.format(index, output_len)), 'w', encoding='utf-8') as of:
            for p, sent, score, topk_idx in zip(predict, predict_sents, attnScore, topkPred):
                of.write('{0}\t{1}\t{2}\t{3}\n'.format(sent, p, tuple(
                    s.tolist() for s in score), tuple(ti.tolist() for ti in topk_idx)))

    # return scores['rouge-2']['f']
    return [scores]


def evalModel(model: nn.Module, summarizer, evalData, output_len: int = 1, prefix: str = 'dev', postfix: str = '', stripping_mode: str = opt.stripping_mode, specifyEpoch: int = -1):
    """
    Output length is used for debug purpose, meant to output more sentence at once.
    Make sure the beam_size is greater or equal to n_best_size.
    (The code is warped in the if output_len > 1)


    TODO: involve log-linear model in decoding?!
    """
    global evalModelCount

    if specifyEpoch <= 0:
        specifyEpoch = evalModelCount

    predict, gold, predict_sents, attnScore, topkPred, gold_sents = getLabel(
        summarizer, evalData, output_len, threshold=opt.threshold, stripping_mode=stripping_mode, isEval=True)

    scores_total = compute_selection_acc(gold, predict)
    scores_hit1 = compute_selection_acc(gold, predict, hit1mode=True)
    scores_metrics = compute_metric(gold, predict)
    # import ipdb; ipdb.set_trace()
    # scores_rouge = compute_rouge(gold, predict)
    # import ipdb; ipdb.set_trace()
    # scores_rouge_raw = compute_rouge_raw(gold_sents, predict_sents)
    scores_bleu_raw = compute_bleu_raw(
        gold_sents, predict_sents, gold, predict)

    if postfix:
        postfix = '.' + postfix

    with open(os.path.join(opt.save_path, '{1}{2}.out.{0}'.format(specifyEpoch, prefix, postfix)), 'w', encoding='utf-8') as of:
        for p, sent in zip(predict, predict_sents):
            of.write('{0}\t{1}'.format(sent, p) + '\n')

    if output_len > 1:

        with open(os.path.join(opt.save_path, '{1}{2}.{3}_n_out.{0}'.format(specifyEpoch, prefix, postfix, output_len)), 'w', encoding='utf-8') as of:
            for p, sent, score, topk_idx in zip(predict, predict_sents, attnScore, topkPred):
                of.write('{0}\t{1}\t{2}\t{3}\n'.format(sent, p, tuple(
                    s.tolist() for s in score), tuple(ti.tolist() for ti in topk_idx)))
                # note that, the topk_idx is not necessary be the same as idx,
                # since beam search finds the highest score of a single route,
                # so it is possible that it didn't select the "max attention" score index on a step

    # return [scores_total, scores_hit1, scores_metrics, scores_rouge, scores_rouge_raw, scores_bleu_raw]
    return [scores_total, scores_hit1, scores_metrics, scores_bleu_raw]


def gen_pointer_mask(target_lengths, batch_size, max_step):
    res = torch.ByteTensor(batch_size, max_step).fill_(1)
    for idx, ll in enumerate(target_lengths.data[0]):
        if ll == max_step:
            continue
        res[idx][ll:] = 0
    res = res.float()
    return res


def getDevNLLLoss(model: nn.Module, validData):
    global totalBatchCount

    loss_crit = NMTCriterion()

    total_docs = 0
    total_reg_loss = 0

    for i in range(len(validData)):
        batch = validData[i][0]

        doc_sent_scores, doc_sent_mask = model(batch)

        oracle_targets = batch[2][0]
        nll_loss = compute_nll_loss(
            doc_sent_scores, oracle_targets, doc_sent_mask, loss_crit)

        report_reg_loss = nll_loss.item()
        num_of_docs = doc_sent_mask.size(0)

        total_reg_loss += report_reg_loss
        total_docs += num_of_docs

    return total_reg_loss, total_reg_loss / total_docs


def determine_good_data(max_length, doc_sent_scores, mode, significant_value):
    assert mode in ['threshold', 'magnitude']
    new_good_data = []
    # (1, batch_size, max_length)
    doc_sent_scores = doc_sent_scores.squeeze(0)
    max_length = max_length.squeeze(0)
    # (batch_size, max_length)
    if mode == 'threshold':
        values, _ = doc_sent_scores.topk(k=1, dim=1, largest=True)
        # (batch_size, 1)
        for val, length in zip(values, max_length):
            if val > significant_value:
                # if val > significant_value / length:
                new_good_data.append(True)
            else:
                new_good_data.append(False)

    elif mode == 'magnitude':
        values, _ = doc_sent_scores.topk(k=2, dim=1, largest=True, sorted=True)
        # (batch_size, 2)
        for val, length in zip(values, max_length):
            if val[0] / val[1] > significant_value:
                new_good_data.append(True)
            else:
                new_good_data.append(False)

    return new_good_data


def trainModel(model: nn.Module, summarizer, trainData, validData, dataset, optim, save_best_metrics: bool = False,
               log_linear_model: nn.Module = None, log_linear_optim=None, testData=None):
    """
        main training function
        when use iterative learning, better set false to save_best_metrics,
        because the dev set labels are not changed, so evaluation become no reference value
    """
    global evalModelCount

    already_relabeled = False
    early_stop = False

    if opt.IL_with_KLDivLoss:
        assert not opt.extra_shuffle, "Shouldn't shuffle data while IL with KLDivLoss"
        previous_round_student_distribution = None
        this_round_student_distribution = None

    if opt.start_update_good_data_set >= 0:
        assert not opt.extra_shuffle, "Shouldn't shuffle data while update good data set"
        assert trainData.good_paper is not None
        previous_round_good_data_set = None
        if opt.keep_initial_good_data_unchange:
            initial_good_data_set = trainData.good_paper

    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format(
        'train' if model.training else 'eval'))

    if opt.relabel_epoch < opt.stop_em_and_loglinear_after - 1:
        logger.warning(
            'Relabel data will not work during EM and will slow the process.')

    # define criterion of each GPU
    if log_linear_model is not None:
        assert log_linear_optim is not None
        assert opt.max_decode_step == 1  # assume step is 1 during training
        log_linear_model.train()
        logger.warning("Set log-linear model to {0} mode".format(
            'train' if log_linear_model.training else 'eval'))
        # size_average & reduce arguments are deprecated
        # regression_crit = nn.KLDivLoss(size_average=False, reduce=False)
        regression_crit = nn.KLDivLoss(reduction='none')
        EM_TRAIN_STUDENT = True  # for log-linear model EM training
    else:
        assert log_linear_optim is None

    loss_crit = NMTCriterion()

    # logger.info('Eval on the untrained model')
    # model.eval()
    # valid_bleu = evalTrainData(model, summarizer, trainData)
    # model.train()

    def saveModel(model, optim, filename: str = 'model', data=None, metric=None):
        """ save the model, if metric is not None then save with the name of the metrics value """
        model_state_dict = model.module.state_dict() if len(
            opt.gpus) > 1 else model.state_dict()
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if 'generator' not in k}
        #  (4) drop a checkpoint

        if data is not None:
            relabeled_oracle = data.oracle
        else:
            relabeled_oracle = None

        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            # TODO: store state_dict of optimizer instead of object
            'optim': optim,
            'relabeled_oracle': relabeled_oracle,
            'eval_model_count': evalModelCount,  # TODO: this should be local variable
            'total_batch_count': totalBatchCount,  # TODO: this should be local variable
        }
        save_model_path = filename
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_devRouge_{1}_epoch_{2}.pt'.format(
                save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_epoch_{1}.pt'.format(
                save_model_path, epoch))

    def trainEpoch(epoch: int, teach_student: bool = True):
        """ detail of each training epoch """

        if opt.extra_shuffle and epoch > opt.curriculum and not opt.IL_with_KLDivLoss:
            # NOTE: this might cause the labeling output inconsistent for analyzing, better diable this function
            # NOTE: when using IL with KLDivloss, we need the batch order is the same
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_reg_loss = total_point_loss = total_docs = total_points = 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            """
            (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)
            """
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]

            # https://discuss.pytorch.org/t/what-step-backward-and-zero-grad-do/33301
            model.zero_grad()
            if log_linear_model is not None and opt.stop_em_and_loglinear_after > epoch:
                log_linear_model.zero_grad()

            # regression loss
            # gold_sent_rouge_scores = batch[5][0]  # (batch*step, doc_len)
            # reg_loss = regression_loss(doc_sent_scores, gold_sent_rouge_scores, doc_sent_mask, regression_crit)
            if log_linear_model is not None and opt.stop_em_and_loglinear_after > epoch:
                # Teacher-student training & co-teaching

                if teach_student:
                    # set log-linear model to eval mode
                    log_linear_model.eval()
                else:
                    model.eval()

                # Make sure models are correctly set in train or eval mode
                # https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744
                if teach_student:
                    assert model.training
                    assert not log_linear_model.training
                else:
                    assert log_linear_model.training
                    assert not model.training

                # (step, batch, doc_len), (batch, doc_len)
                doc_sent_scores, doc_sent_mask = model(batch)

                gold_sent_loglinear_scores, _ = log_linear_model(batch)
                max_point_step = batch[2][1].max().item()

                if opt.start_update_good_data_set >= 0 and epoch >= opt.start_update_good_data_set:
                    # update good data set
                    # opt.start_update_good_data_set < 0 means disable
                    # TODO: keep the "original/initial good data"
                    new_good_data = determine_good_data(batch[0][2], doc_sent_scores,
                                                        opt.significant_criteria, opt.significant_value)

                    start, end = batch[8]
                    for i in range(start, end):
                        if opt.keep_initial_good_data_unchange:
                            trainData.good_paper[i] = new_good_data[i - start] | initial_good_data_set[i]
                        else:
                            trainData.good_paper[i] = new_good_data[i - start]

                if epoch <= opt.use_good_data_only_before:
                    good_paper_marker_batch, _ = batch[5]
                    for j in range(doc_sent_mask.shape[0]):
                        if not good_paper_marker_batch[j]:
                            doc_sent_mask[j, :] = 1.0

                elif opt.add_all_bad_data_after > epoch >= opt.use_good_data_only_before:
                    assert opt.significant_criteria == 'random_ratio'
                    _, bad_indices = batch[5]
                    percent = (epoch - opt.use_good_data_only_before) / \
                        (opt.add_all_bad_data_after -
                         opt.use_good_data_only_before)
                    nums = int((1 - percent) * len(bad_indices))

                    bad_sample_to_exclude = random.sample(bad_indices, nums)

                    for j in bad_sample_to_exclude:
                        if j < doc_sent_mask.shape[0]:
                            doc_sent_mask[j, :] = 1.0

                reg_loss = regression_loss(
                    doc_sent_scores, gold_sent_loglinear_scores, doc_sent_mask, regression_crit, teach_student)

                # switch back to train mode (default)
                if teach_student:
                    # set log-linear model back to train mode
                    log_linear_model.train()
                else:
                    model.train()

                loss = reg_loss
                # report_reg_loss = reg_loss.data[0]
                report_reg_loss = reg_loss.item()
            else:
                # Self-training & Iterative Labeling
                assert opt.significant_criteria not in ['threshold', 'magnitude'], \
                    "Should set '-significant_criteria' random_ratio mode"

                if opt.IL_with_KLDivLoss and opt.stop_em_and_loglinear_after <= epoch:
                    assert len(previous_round_student_distribution) == len(
                        trainData)
                    doc_sent_scores, doc_sent_mask = model(batch)

                    if epoch <= opt.use_good_data_only_before:
                        # TODO: bad_indices is used for future extension of incrementally add data
                        good_paper_marker_batch, bad_indices = batch[5]
                        for j in range(doc_sent_mask.shape[0]):
                            if not good_paper_marker_batch[j]:
                                doc_sent_mask[j, :] = 1.0

                    elif opt.add_all_bad_data_after > epoch >= opt.use_good_data_only_before:
                        assert opt.significant_criteria == 'random_ratio'
                        good_paper_marker_batch, bad_indices = batch[5]
                        percent = (epoch - opt.use_good_data_only_before) / \
                            (opt.add_all_bad_data_after -
                             opt.use_good_data_only_before)
                        nums = int((1 - percent) * len(bad_indices))

                        bad_sample_to_exclude = random.sample(
                            bad_indices, nums)

                        for j in bad_sample_to_exclude:
                            if j < doc_sent_mask.shape[0]:
                                doc_sent_mask[j, :] = 1.0

                    reg_loss = regression_loss(doc_sent_scores, previous_round_student_distribution[i],
                                               doc_sent_mask, regression_crit, True)
                    loss = reg_loss
                    report_reg_loss = reg_loss.item()
                else:
                    # NLLoss Case & Original Relabel case

                    # (step, batch, doc_len), (batch, doc_len)
                    doc_sent_scores, doc_sent_mask = model(batch)
                    oracle_targets = batch[2][0]

                    if epoch <= opt.use_good_data_only_before:
                        _, bad_indices = batch[5]

                        # nll loss
                        nll_loss = compute_nll_loss(
                            doc_sent_scores, oracle_targets, doc_sent_mask, loss_crit, bad_indices)

                    elif opt.add_all_bad_data_after > epoch >= opt.use_good_data_only_before:
                        assert opt.significant_criteria == 'random_ratio'
                        _, bad_indices = batch[5]
                        percent = (epoch - opt.use_good_data_only_before) / \
                            (opt.add_all_bad_data_after -
                             opt.use_good_data_only_before)
                        nums = int((1 - percent) * len(bad_indices))

                        bad_sample_to_exclude = random.sample(
                            bad_indices, nums)

                        for j in bad_sample_to_exclude:
                            if j < doc_sent_mask.shape[0]:
                                doc_sent_mask[j, :] = 1.0

                        # nll loss
                        nll_loss = compute_nll_loss(
                            doc_sent_scores, oracle_targets, doc_sent_mask, loss_crit, bad_sample_to_exclude)

                    else:

                        nll_loss = compute_nll_loss(
                            doc_sent_scores, oracle_targets, doc_sent_mask, loss_crit)

                    loss = nll_loss
                    report_reg_loss = nll_loss.item()

            if opt.IL_with_KLDivLoss:
                this_round_student_distribution.append(doc_sent_scores)

            report_point_loss = 0
            num_of_docs = doc_sent_mask.size(0)
            num_of_pointers = 0
            total_reg_loss += report_reg_loss
            total_point_loss += report_point_loss
            total_docs += num_of_docs
            total_points += num_of_pointers

            # update the parameters
            loss.backward()

            if log_linear_model is not None and not teach_student and opt.stop_em_and_loglinear_after > epoch:
                log_linear_optim.step()
            else:
                optim.step()

            if i % opt.log_interval == -1 % opt.log_interval:
                logger.info(
                    "Epoch %2d, %6d/%5d/%5d; reg_loss: %6.2f; docs: %5d; avg_reg_loss: %6.2f; %6.0f s elapsed" %
                    (epoch, totalBatchCount, i + 1, len(trainData),
                     report_reg_loss,
                     num_of_docs,
                     report_reg_loss / num_of_docs,
                     time.time() - start))

                writer.add_scalar('train/reg_loss',
                                  report_reg_loss, totalBatchCount)
                writer.add_scalar('train/avg_reg_loss',
                                  report_reg_loss / num_of_docs, totalBatchCount)

                start = time.time()

            # TODO: NLL Loss diagram on Dev Set

            if validData is not None and len(validData) > 0 and totalBatchCount % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format(
                    'train' if model.training else 'eval'))
                # valid_score = evalModel(
                #     model, summarizer, validData, output_len=opt.output_len)
                # logger.info(valid_score)
                dev_loss = getDevNLLLoss(model, validData)
                writer.add_scalar('dev/reg_loss',
                                  dev_loss[0], totalBatchCount)
                writer.add_scalar('dev/avg_reg_loss',
                                  dev_loss[1], totalBatchCount)
                model.train()
                logger.warning("Set model to {0} mode".format(
                    'train' if model.training else 'eval'))
            #     valid_bleu = all_valid_bleu[0]
            #     logger.info('Validation Score: %g' % (valid_bleu * 100))
            #     logger.info('Best Metric: %g' % (optim.best_metric * 100))
                # writer.add_scalar('dev/precision',
                #                   valid_score[2]['p'], totalBatchCount)
                # writer.add_scalar('dev/recall',
                #                   valid_score[2]['r'], totalBatchCount)
                # writer.add_scalar('dev/f1',
                #                   valid_score[2]['f1'], totalBatchCount)

                # TODO: set min loss as optim.best_metric?!

            #     if save_best_metrics and valid_bleu >= optim.best_metric:
            #         # only save model when it get the best metric
            #         saveModel(valid_bleu)
            #     optim.updateLearningRate(valid_bleu, epoch)

        return total_reg_loss / total_docs

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')

        assert not early_stop

        if opt.IL_with_KLDivLoss:
            this_round_student_distribution = []

        # Logging
        if log_linear_model is not None and epoch == opt.start_train_with_em:
            logger.info('Start training with EM (teach-student).')

        if log_linear_model is not None and epoch == opt.stop_em_and_loglinear_after:
            logger.info('Stop EM and back to normal.')

            if not opt.IL_with_KLDivLoss:
                logger.info(
                    'Training data oracle will be relabeled at the end of this epoch.')
            else:
                logger.info(
                    'Training data distribution will be stored at the end of this epoch.')

        if opt.significant_criteria == 'random_ratio':
            if epoch == opt.use_good_data_only_before:
                logger.info('Start adding other data')

            if epoch == opt.add_all_bad_data_after:
                logger.info('All data are added')
        else:
            # threshold or magnitude
            assert epoch < opt.stop_em_and_loglinear_after, \
                "You should use teach-student or co-teaching during this mode."

            if epoch == opt.start_update_good_data_set:
                logger.info(
                    f'The good data set will be updated during training with the {opt.significant_criteria} mode now.')

        # TODO: determine teach_student first or after
        if log_linear_model is not None and opt.stop_em_and_loglinear_after > epoch >= opt.start_train_with_em:
            # train log-linear with student's output
            logger.info('Student teach teacher')
            train_reg_loss = trainEpoch(epoch, teach_student=False)

        #  (1) train for one epoch on the training set
        logger.info('Teacher teach student or self-learning')
        train_reg_loss = trainEpoch(epoch)

        # for iterative labeling
        # (will be ignored when training log-linear model with EM)
        if epoch >= opt.relabel_epoch and not opt.IL_with_KLDivLoss:
            logger.info('Relabeling training data')
            if not opt.relabel_once or (opt.relabel_once ^ already_relabeled):
                RelabelTrainData(summarizer, trainData,
                                 output_len=opt.output_len)
                already_relabeled = True

        if opt.IL_with_KLDivLoss:
            logger.info(
                'Update previous round student distribution for IL with KLDivLoss.')
            previous_round_student_distribution = this_round_student_distribution

        if opt.start_update_good_data_set >= 0 and epoch >= opt.start_update_good_data_set - 1:
            # opt.start_update_good_data_set < 0 means disable
            logger.info('Update previous round good data set.')
            if previous_round_good_data_set is not None:
                diff_cnt = sum([prev != curr for prev, curr in zip(
                    previous_round_good_data_set, trainData.good_paper)])
                diff_percent = diff_cnt / len(previous_round_good_data_set)
                writer.add_scalar('train/diff_data_percent',
                                  diff_percent, epoch)
                writer.add_scalar('train/good_data_percent', sum(
                    trainData.good_paper) / len(previous_round_good_data_set), epoch)
                if opt.early_stop_change_percent >= 0 and diff_percent < opt.early_stop_change_percent:
                    logger.info('Early stop at the end of this epoch')
                    early_stop = True
            previous_round_good_data_set = trainData.good_paper.copy()
        elif epoch < opt.start_update_good_data_set:
            # TODO: can move to the "else" of "if previous_round_good_data_set is not None"
            good_data_percent = sum(trainData.good_paper) / \
                len(trainData.good_paper)
            logger.info(
                f'Showing the good data percent before update: {good_data_percent}')
            writer.add_scalar('train/good_data_percent',
                              good_data_percent, epoch)

        model.eval()
        evalTrainData(model, summarizer, trainData,
                      epoch, output_len=opt.output_len)
        if opt.eval_test_during_train:
            evalModelCount += 1
            scores = evalModel(model, summarizer, testData,
                               prefix='test_during_train')
            logger.info(
                'Evaluate score: (accuracy) total / hit@1, {precision, recall, f1 score} (sentence-level), {corpus_bleu, avg_sent_bleu}')
            logger.info(scores)
            writer.add_scalar('train/precision', scores[2]['p'], epoch)
            writer.add_scalar('train/recall', scores[2]['r'], epoch)
            writer.add_scalar('train/f1', scores[2]['f1'], epoch)
            writer.add_scalar('train/corpus_bleu', scores[3]['corpus_bleu'], epoch)
            writer.add_scalar('train/avg_sent_bleu', scores[3]['avg_sent_bleu'], epoch)

            if log_linear_model is not None and opt.stop_em_and_loglinear_after > epoch:
                # evaluate log-linear model only when EM is on
                log_linear_model.eval()
                scores = evalLoglinearModel(log_linear_model, testData, evalModelCount=epoch,
                                            prefix='test_loglinear_during_train', output_len=opt.output_len)
                logger.info(
                    'Evaluate log-linear score: (accuracy) total / hit@1, {precision, recall, f1 score} (sentence-level), {corpus_bleu, avg_sent_bleu}')
                logger.info(scores)
                logger.info('Rule used: %s' % log_linear_model.rules_used)
                logger.info('Rule weights: %s' %
                            log_linear_model.gamma.weight.data)
                writer.add_scalar('train_loglinear/precision',
                                  scores[2]['p'], epoch)
                writer.add_scalar('train_loglinear/recall',
                                  scores[2]['r'], epoch)
                writer.add_scalar('train_loglinear/f1', scores[2]['f1'], epoch)
                writer.add_scalar('train_loglinear/corpus_bleu', scores[3]['corpus_bleu'], epoch)
                writer.add_scalar('train_loglinear/avg_sent_bleu', scores[3]['avg_sent_bleu'], epoch)
                if not opt.section_embedding:
                    for i, rule in enumerate(log_linear_model.rules_used):
                        writer.add_scalar(
                            f'train_loglinear/W_{rule}', log_linear_model.gamma.weight.data.tolist()[0][i], epoch)

                log_linear_model.train()
        model.train()

        logger.info('Train regression loss: %g' % train_reg_loss)
        if opt.dump_epoch_checkpoint:
            logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
            saveModel(model, optim, 'model', trainData)
            if log_linear_model is not None:
                saveModel(log_linear_model, log_linear_optim,
                          'log_linear_model')

        if early_stop:
            logger.info('Early stopping')
            return


def main():
    if not opt.online_process_data:
        raise Exception(
            'This code does not use preprocessed .pt pickle file. It has some issues with big files.')
        # dataset = torch.load(opt.data)
    else:
        import onlinePreprocess
        onlinePreprocess.seq_length = opt.max_sent_length
        onlinePreprocess.max_doc_len = opt.max_doc_len
        onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
        onlinePreprocess.norm_lambda = opt.norm_lambda
        from onlinePreprocess import prepare_data_online
        # TODO: some NMT stuff we don't use, consider to clean it up...
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                      opt.train_src_rouge, opt.train_src_section, opt.drop_too_short, opt.drop_too_long)

    # TODO: some NMT stuff we don't use, consider to clean it up...
    trainData = neusum.Dataset(dataset['train']['src'], dataset['train']['src_raw'], dataset['train']['tgt'],
                               dataset['train']['oracle'], dataset['train']['src_rouge'], dataset[
                                   'train']['src_section'], dataset['train']['src_section_raw'],
                               opt.batch_size, opt.max_doc_len, opt.gpus, dataset[
                                   'train']['bert_annotation'],
                               good_patterns=loglinear.Config.Keyword[opt.qtype], use_good=True)

    dicts = dataset['dicts']
    # logger.info(' * vocabulary size. source = %d; target = %d' %
    #             (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * vocabulary size. source = %d' %
                (dicts['src'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building model...')

    sent_encoder = neusum.Models.Encoder(opt, dicts['src'])
    doc_encoder = neusum.Models.DocumentEncoder(opt)
    pointer = neusum.Models.Pointer(opt)
    if opt.dec_init == "simple":
        decIniter = neusum.Models.DecInit(opt)
    elif opt.dec_init == "att":
        decIniter = neusum.Models.DecInitAtt(opt)
    else:
        raise ValueError(
            'Unknown decoder init method: {0}'.format(opt.dec_init))

    model = neusum.Models.NMTModel(
        sent_encoder, doc_encoder, pointer, decIniter)
    summarizer = neusum.Summarizer(opt, model, dataset)
    log_linear_model = None
    log_linear_optim = None

    if opt.freeze_word_vecs_enc:
        logger.warning('Not updating encoder word embedding.')

    if opt.continue_training:
        # Assume same log-linear model checkpoint at the same place
        model_path = glob(os.path.join(opt.save_path, 'model_epoch_*.pt'))
        if model_path:
            opt.log_linear_save_path = opt.save_path
            logger.info('Try to find checkpoints to recover')
            model_selected = sorted(model_path, key=os.path.getmtime)[-1]
            logger.info('Loading from the latest model "%s"' % model_selected)
            model_checkpoint = torch.load(model_selected)
            logger.info('Load model weights.')
            model.load_state_dict(model_checkpoint['model'])
            logger.info('Load optimizer.')
            optim = model_checkpoint['optim']
            logger.info('Load epoch.')
            opt.start_epoch = model_checkpoint['epoch'] + 1
            if opt.start_epoch > opt.epochs:
                logger.warning("The checkpoint is already the latest checkpoint. Don't need training anymore, otherwise larger the -epochs argument.")
            else:
                logger.info(f'Model will continue training on epoch {opt.start_epoch}')

            if model_checkpoint.get('relabeled_oracle') is not None:
                logger.info('Load relabeled oracle.')
                trainData.oracle = model_checkpoint['relabeled_oracle']
            # TODO: this should be local variable
            if model_checkpoint.get('eval_model_count') is not None:
                logger.info('Load eval model count')
                global evalModelCount
                evalModelCount = model_checkpoint['eval_model_count']
            if model_checkpoint.get('total_batch_count') is not None:
                logger.info('Load total batch count')
                global totalBatchCount
                totalBatchCount = model_checkpoint['total_batch_count']
        else:
            logger.info('No checkpoint found. Train new one.')

            logger.info('Initial model weights.')
            for pr_name, p in model.named_parameters():
                logger.info(pr_name)
                # p.data.uniform_(-opt.param_init, opt.param_init)
                if p.dim() == 1:
                    # p.data.zero_()
                    p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
                else:
                    nn.init.xavier_normal_(p, math.sqrt(3))
                    # xavier_uniform(p)

            logger.info('Load pre-trained vectors.')
            sent_encoder.load_pretrained_vectors(opt, logger)

            logger.info('Initial optimizer.')
            optim = neusum.Optim(
                opt.optim, opt.learning_rate,
                max_grad_norm=opt.max_grad_norm,
                max_weight_value=opt.max_weight_value,
                lr_decay=opt.learning_rate_decay,
                start_decay_at=opt.start_decay_at,
                decay_bad_count=opt.halve_lr_bad_count
            )

            optim.set_parameters(model.parameters())

            if opt.enable_log_linear:
                log_linear_optim = neusum.Optim(
                    opt.optim, opt.loglinear_learning_rate,
                    max_grad_norm=opt.max_grad_norm,
                    max_weight_value=opt.max_weight_value,
                    lr_decay=opt.learning_rate_decay,
                    start_decay_at=opt.start_decay_at,
                    decay_bad_count=opt.halve_lr_bad_count
                )

    # TODO: store state_dict of optimizer instead of object
    optim.set_parameters(model.parameters())

    # Load log-linear model
    if opt.log_linear_save_path and opt.enable_log_linear:
        # Find the latest model to load
        model_path = glob(os.path.join(
            opt.log_linear_save_path, 'log_linear_model_epoch_*.pt'))
        if not model_path:
            raise ValueError("Can't find log linear model %s" %
                             os.path.join(opt.log_linear_save_path, 'log_linear_model_epoch_*.pt'))

        log_linear_model_selected = sorted(
            model_path, key=os.path.getmtime)[-1]
        logger.info('Loading from the latest log-linear model "%s"' %
                    log_linear_model_selected)

    if len(opt.gpus) >= 1:
        model.cuda()
        if opt.enable_log_linear:
            log_linear_model = loglinear.model.LogLinear(use_gpu=True).cuda()
            log_linear_model.set_rules(opt.position_weight, opt.keyword_weight,
                                       loglinear.Config.Keyword[opt.qtype], opt.in_bert_weight,
                                       opt.in_section_weight, loglinear.Config.PossibleSection[opt.qtype],
                                       opt.section_embedding, opt.pre_word_vecs_enc)
            if opt.log_linear_save_path:
                logger.info('Set log-linear weights with checkpoint.')
                log_linear_checkpoint = torch.load(log_linear_model_selected)
                log_linear_model.load_state_dict(
                    log_linear_checkpoint['model'])
                # TODO: store state_dict of optimizer instead of object
                log_linear_optim = log_linear_checkpoint['optim']
                log_linear_model.cuda()
            else:
                # TODO: store state_dict of optimizer instead of object
                log_linear_optim.set_parameters(log_linear_model.parameters())
    else:
        model.cpu()
        if opt.enable_log_linear:
            log_linear_model = loglinear.model.LogLinear(use_gpu=False)
            log_linear_model.set_rules(opt.position_weight, opt.keyword_weight,
                                       loglinear.Config.Keyword[opt.qtype], opt.in_bert_weight,
                                       opt.in_section_weight, loglinear.Config.PossibleSection[opt.qtype],
                                       opt.section_embedding, opt.pre_word_vecs_enc)
            if opt.log_linear_save_path:
                logger.info('Set log-linear weights with checkpoint.')
                log_linear_checkpoint = torch.load(
                    log_linear_model_selected, map_location=torch.device('cpu'))
                log_linear_model.load_state_dict(
                    log_linear_checkpoint['model'])
                # TODO: store state_dict of optimizer instead of object
                log_linear_optim = log_linear_checkpoint['optim']
            else:
                # TODO: store state_dict of optimizer instead of object
                log_linear_optim.set_parameters(log_linear_model.parameters())
    
    # TODO: store state_dict of optimizer instead of object
    log_linear_optim.set_parameters(log_linear_model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        logger.info('Loading dev set')
        validData = load_dev_data(
            summarizer, opt.dev_input_src, opt.dev_ref, opt.dev_input_src_section, opt.drop_too_short, opt.drop_too_long)
    testData = None
    if opt.test_input_src and opt.test_ref:
        logger.info('Loading test set')
        testData = load_dev_data(
            summarizer, opt.test_input_src, opt.test_ref, opt.test_input_src_section, opt.drop_too_short, opt.drop_too_long, opt.test_bert_annotation)

    trainModel(model, summarizer, trainData, validData, dataset,
               optim, log_linear_model=log_linear_model, log_linear_optim=log_linear_optim,
               testData=testData)


if __name__ == "__main__":
    main()
    writer.export_scalars_to_json(f"{opt.save_path}/all_scalars.json")
    writer.close()
