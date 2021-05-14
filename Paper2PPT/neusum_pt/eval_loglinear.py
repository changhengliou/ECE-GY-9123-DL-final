# Currently deprecated
# because there might be a data leak during test
# (because the oracle of the test data is built by gold label instead of BERT annotation)

import argparse
import logging
import time
import os
from glob import glob
import torch
import neusum
import loglinear
import xargs
from typing import List, Tuple
import nltk

parser = argparse.ArgumentParser(
    description='Test for the log-linear model')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
if len(logging.root.handlers) == 1:  # only default handler
    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.test.log.txt'
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
        torch.cuda.set_device(opt.gpus[0])
        logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

    logger.info('My seed is {0}'.format(torch.initial_seed()))


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


def compute_bleu_raw(gold_sents: List[Tuple[str]], predict_sents: List[Tuple[str]], gold: List[Tuple[int]], predict: List[Tuple[int]]):
    """ copy from train.py """
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


def getLabelFromLogLinearModel(model, batch, output_len: int = 1, threshold: float = 0.0001, stripping_mode: str = 'topk'):
    # assert stripping_mode in ('none', 'normal', 'magnitude', 'topk')
    assert stripping_mode in ('none', 'diff', 'topk')
    # TODO: normal, magnitude mode & use topk, diff in the condition
    doc_sent_scores, doc_sent_mask = model(batch)
    scores = torch.softmax(doc_sent_scores, dim=1)
    pred_scores, pred_labels = torch.max(scores, dim=1)
    if output_len == 1:
        return pred_labels, pred_scores, None, None
    else:
        topk_pred_scores, topk_pred_labels = scores.topk(
            k=output_len, dim=1, largest=True, sorted=True)
        return pred_labels, pred_scores, topk_pred_scores, topk_pred_labels


def evalModel(model, testData, prefix: str = 'test_loglinear', evalModelCount: int = 1, output_len: int = 5, stripping_mode: str = 'topk', postfix: str = ''):

    predict, gold, predict_sents, predictScore, gold_sents = [], [], [], [], []
    good_or_bad, topk_predict_idx, topk_predict_sents, topk_predictScore = [], [], [], []
    stripped_topk_predict_idx, stripped_topk_predict_sents, stripped_topk_predictScore = [], [], []

    for i in range(len(testData)):
        batch = testData[i][0]

        pred, predScore, topk_pred_scores, topk_pred_labels = getLabelFromLogLinearModel(
            model, batch, output_len=output_len, threshold=opt.threshold)

        gold_ids, gold_lengths = batch[2][0].tolist(
        ), batch[2][1].view(-1).tolist()

        src_raw = batch[1][0]
        src_section_raw = batch[7][0]
        for b in range(len(src_raw)):

            idx = pred[b]
            if idx >= len(src_raw[b]):
                continue

            predictScore.append((predScore[b].item(),))
            predict_sents.append(((src_raw[b][idx], src_section_raw[b][idx]),))
            predict.append((idx.item(),))
            gold.append(tuple(gold_ids[b][:gold_lengths[b]]))
            gold_sents.append(
                tuple((src_raw[b][idx], src_section_raw[b][idx]) for idx in gold[-1])
            )
            good_or_bad.append(batch[5][0][b])

            if output_len > 1:
                topk_predict_sents.append(tuple((src_raw[b][idx.item()], src_section_raw[b][idx.item()]) for idx in topk_pred_labels[b]
                                                if idx < len(src_raw[b])))
                topk_predict_idx.append(tuple(idx.item() for idx in topk_pred_labels[b]
                                                if idx < len(src_raw[b])))
                topk_predictScore.append(tuple(idx.item() for idx in topk_pred_scores[b]
                                                if idx < len(src_raw[b])))
            
   
                # diff mode
                for j in range(1, len(topk_pred_scores[b])):
                    if topk_pred_scores[b][j - 1] - topk_pred_scores[b][j] > opt.threshold:
                        break
                
                j = min(j, output_len)

                stripped_topk_predict_sents.append(tuple((src_raw[b][idx.item()], src_section_raw[b][idx.item()])
                                                   for idx in topk_pred_labels[b][:j] if idx < len(src_raw[b])))
                stripped_topk_predict_idx.append(tuple(idx.item() for idx in topk_pred_labels[b][:j]
                                                if idx < len(src_raw[b])))
                stripped_topk_predictScore.append(tuple(idx.item() for idx in topk_pred_scores[b][:j]
                                                if idx < len(src_raw[b])))

    scores_total = compute_selection_acc(gold, predict)
    scores_hit1 = compute_selection_acc(gold, predict, hit1mode=True)
    scores_metrics = compute_metric(gold, predict)
    scores_bleu_raw = compute_bleu_raw(gold_sents, predict_sents, gold, predict)

    with open(os.path.join(opt.save_path, '{1}.out.{0}'.format(evalModelCount, prefix)), 'w', encoding='utf-8') as of:
        for p, sent, score in zip(predict, predict_sents, predictScore):
            of.write('{0}\t{1}\t{2}'.format(sent, p, score) + '\n')
    
    topk_scores_total = None
    topk_scores_hit1 = None
    topk_scores_metrics = None
    topk_scores_bleu_raw = None
    if output_len > 1:
        topk_scores_total = compute_selection_acc(gold, topk_predict_idx)
        topk_scores_hit1 = compute_selection_acc(gold, topk_predict_idx, hit1mode=True)
        topk_scores_metrics = compute_metric(gold, topk_predict_idx)
        topk_scores_bleu_raw = compute_bleu_raw(gold_sents, topk_predict_sents, gold, topk_predict_idx)

        with open(os.path.join(opt.save_path, '{1}{2}_{3}.{4}_n_out.{0}'.format(evalModelCount, prefix, postfix, stripping_mode, output_len)), 'w', encoding='utf-8') as of:
            for p, sent, score, topk_sent, topk_idx, good_bad in zip(predict, predict_sents, topk_predictScore, topk_predict_sents, topk_predict_idx, good_or_bad):
                of.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(good_bad, sent, p, topk_sent, topk_idx, score))
        
        stripped_topk_scores_total = compute_selection_acc(gold, stripped_topk_predict_idx)
        stripped_topk_scores_hit1 = compute_selection_acc(gold, stripped_topk_predict_idx, hit1mode=True)
        stripped_topk_scores_metrics = compute_metric(gold, stripped_topk_predict_idx)
        stripped_topk_scores_bleu_raw = compute_bleu_raw(gold_sents, stripped_topk_predict_sents, gold, stripped_topk_predict_idx)

        with open(os.path.join(opt.save_path, '{1}{2}_{3}.{4}_stripped_n_out.{0}'.format(evalModelCount, prefix, postfix, stripping_mode, output_len)), 'w', encoding='utf-8') as of:
            for p, sent, score, topk_sent, topk_idx, good_bad in zip(predict, predict_sents, stripped_topk_predictScore, stripped_topk_predict_sents, stripped_topk_predict_idx, good_or_bad):
                of.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(good_bad, sent, p, topk_sent, topk_idx, score))


    return scores_total, scores_hit1, scores_metrics, scores_bleu_raw, \
           topk_scores_total, topk_scores_hit1, topk_scores_metrics, topk_scores_bleu_raw, \
           stripped_topk_scores_total, stripped_topk_scores_hit1, stripped_topk_scores_metrics, stripped_topk_scores_bleu_raw
            


if __name__ == "__main__":
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
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                      opt.train_src_rouge, opt.train_src_section, opt.drop_too_short, opt.drop_too_long)

    dicts = dataset['dicts']

    logger.info('Loading checkpoint...')
    LOAD_MODEL = True
    if opt.specific_epoch > 0:
        model_selected = os.path.join(
            opt.save_path, 'log_linear_model_epoch_%s.pt' % opt.specific_epoch)
        logger.info('Loading from the specific epoch checkpoint "%s"' %
                    model_selected)
    else:
        # Find the latest model to load
        model_path = glob(os.path.join(opt.save_path, '*.pt'))
        if model_path:
            # make sure not load the pointer model
            model_selected = None
            for candidate in reversed(sorted(model_path, key=os.path.getmtime)):
                if 'log_linear_model_epoch' in candidate:
                    model_selected = candidate
                    break
            assert model_selected is not None
            logger.info('Loading from the latest model "%s"' % model_selected)
        else:
            LOAD_MODEL = False
            logger.info("Can't find model %s" %
                        os.path.join(opt.save_path, '*.pt'))
            logger.info('Use manual rule.')

    if LOAD_MODEL:
        if opt.gpus:
            checkpoint = torch.load(model_selected)
        else:
            checkpoint = torch.load(
                model_selected, map_location=torch.device('cpu'))

        logger.info('\tprevious training epochs: %d' % checkpoint['epoch'])

        dicts = checkpoint['dicts']

    # logger.info(' * vocabulary size. source = %d; target = %d' %
    #             (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * vocabulary size. source = %d' %
                (dicts['src'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    # sent_encoder = loglinear.model.SentEncoder(opt, dicts['src'])
    # model = loglinear.model.LogLinear(sent_encoder)
    if opt.gpus:
        model = loglinear.model.LogLinear(use_gpu=True)
    else:
        model = loglinear.model.LogLinear(use_gpu=False)

    # Make sure your checkpoint's weight is match
    # model.set_rules(1.0, 1.0, ['future'], 1.0) # initial gamma tensor
    # model.set_rules(-1.0, 1.0, ['future'], 1.0) # initial gamma tensor
    model.set_rules(opt.position_weight, opt.keyword_weight,
                    loglinear.Config.Keyword[opt.qtype], opt.in_bert_weight,
                    opt.in_section_weight, loglinear.Config.PossibleSection[opt.qtype],
                    opt.section_embedding, opt.pre_word_vecs_enc)

    if LOAD_MODEL:
        model.load_state_dict(checkpoint['model'])

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    from train import load_dev_data
    summarizer = neusum.Summarizer(opt, model, dataset)
    testData = load_dev_data(summarizer, opt.dev_input_src, opt.dev_ref, opt.dev_input_src_section,
                             opt.drop_too_short, opt.drop_too_long, test_bert_annotation=opt.test_bert_annotation)

    model.eval()
    if LOAD_MODEL:
        logger.info('Using checkpoint: %s' % model_selected)
        scores = evalModel(
            model, testData, evalModelCount=checkpoint['epoch'], output_len=opt.output_len, postfix=opt.set_postfix)
    else:
        logger.info('Use manual parameters')
        scores = evalModel(model, testData, output_len=opt.output_len, postfix=opt.set_postfix)
    logger.info('Key hyperparmeters:')
    logger.info('\tKeep Data: [%d, %d)' %
                (opt.drop_too_short, opt.drop_too_long))
    logger.info('\tRule used: %s' % model.rules_used)
    logger.info('\tRule weights: %s' % model.gamma.weight.data.tolist())
    if LOAD_MODEL:
        logger.info('\tTrained epoch: %s' % checkpoint['epoch'])
    logger.info(
        'Evaluate score: (accuracy) total / hit@1, {precision, recall, f1 score} (sentence-level) (First 4 is top-1, Mid 4 is top-%d, and Last 4 is top-%d with stripping)' % (opt.output_len, opt.output_len))
    logger.info(scores)
