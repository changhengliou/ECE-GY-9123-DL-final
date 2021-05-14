import argparse
import logging
import time
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import neusum
import loglinear
import xargs
from eval_loglinear import evalModel
from train import load_dev_data

parser = argparse.ArgumentParser(
    description='Training for the log-linear model')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(
    format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
    torch.cuda.set_device(opt.gpus[0])
    logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))

logger.info('My seed is {0}'.format(torch.initial_seed()))

writer = SummaryWriter(logdir=opt.save_path)

evalModelCount = 0
totalBatchCount = 0


def getLossCriterion():
    """ Simply get NLL Loss now """
    # https://discuss.pytorch.org/t/whats-the-param-size-average-in-loss-it-doesnt-work-as-expected/3976
    crit = nn.NLLLoss(size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def compute_nll_loss(pred_scores, gold_scores, mask, crit):
    """
    :param pred_scores: (batch, doc_len)
    :param gold_scores: (batch*step, doc_len)
    :param mask: (batch, doc_len)
    :param crit:
    :return:
    """
    pred_scores = torch.log_softmax(pred_scores + 1e-8, dim=0)
    gold_scores = gold_scores.view(-1)
    loss = crit(pred_scores, gold_scores)
    return loss


def trainModel(model: nn.Module, trainData: neusum.Dataset, validData, dataset, optim):
    """
    validData is unused now
    """
    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format(
        'train' if model.training else 'eval'))

    loss_crit = getLossCriterion()

    def saveModel(metric=None):
        """ save the model, if metric is not None then save with the name of the metrics value """
        model_state_dict = model.module.state_dict() if len(
            opt.gpus) > 1 else model.state_dict()
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if 'generator' not in k}
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        save_model_path = 'log_linear_model'
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

    def trainEpoch(epoch):
        """ detail of each training epoch """

        if opt.extra_shuffle and epoch > opt.curriculum:
            # NOTE: this might cause the labeling output inconsistent for analyzing, better diable this function
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

            model.zero_grad()
            # (step, batch, doc_len), (batch, doc_len)
            doc_sent_scores, doc_sent_mask = model(batch)

            oracle_targets = batch[2][0]
            nll_loss = compute_nll_loss(
                doc_sent_scores, oracle_targets, doc_sent_mask, loss_crit)

            loss = nll_loss
            report_reg_loss = nll_loss.item()

            report_point_loss = 0
            num_of_docs = doc_sent_mask.size(0)
            num_of_pointers = 0
            total_reg_loss += report_reg_loss
            total_point_loss += report_point_loss
            total_docs += num_of_docs
            total_points += num_of_pointers

            # update the parameters
            loss.backward()
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

        return total_reg_loss / total_docs

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train_reg_loss = trainEpoch(epoch)
        logger.info('Train regression loss: %g' % train_reg_loss)

        model.eval()
        scores = evalModel(model, validData, prefix='test_loglinear_during_pretrain', evalModelCount=epoch, output_len=opt.output_len)
        model.train()
        logger.info('Evaluate score: (accuracy) total / hit@1, {precision, recall, f1 score} (sentence-level)')
        logger.info(scores)
        logger.info('Rule used: %s' % model.rules_used)
        logger.info('Rule weights: %s' % model.gamma.weight.data.tolist())
        writer.add_scalar('train_loglinear/precision', scores[2]['p'], epoch)
        writer.add_scalar('train_loglinear/recall', scores[2]['r'], epoch)
        writer.add_scalar('train_loglinear/f1', scores[2]['f1'], epoch)

        if opt.dump_epoch_checkpoint:
            logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
            saveModel()


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
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                      opt.train_src_rouge, opt.train_src_section, opt.drop_too_short, opt.drop_too_long)

    trainData = neusum.Dataset(dataset['train']['src'], dataset['train']['src_raw'], dataset['train']['tgt'],
                               dataset['train']['oracle'], dataset['train']['src_rouge'], dataset['train']['src_section'], dataset['train']['src_section_raw'],
                               opt.batch_size, opt.max_doc_len, opt.gpus, dataset['train']['bert_annotation'],
                               good_patterns=loglinear.Config.Keyword[opt.qtype], use_good=True)

    dicts = dataset['dicts']
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

    model.set_rules(opt.position_weight, opt.keyword_weight,
                    loglinear.Config.Keyword[opt.qtype], opt.in_bert_weight,
                    opt.in_section_weight, loglinear.Config.PossibleSection[opt.qtype],
                    opt.section_embedding, opt.pre_word_vecs_enc)

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    if opt.freeze_word_vecs_enc:
        logger.warning('Not updating encoder word embedding.')

    # sent_encoder.load_pretrained_vectors(opt, logger)

    optim = neusum.Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )

    optim.set_parameters(model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        summarizer = neusum.Summarizer(opt, model, dataset)
        validData = load_dev_data(summarizer, opt.dev_input_src, opt.dev_ref, opt.dev_input_src_section,
                                  opt.drop_too_short, opt.drop_too_long, test_bert_annotation=opt.test_bert_annotation)

    trainModel(model, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
    writer.export_scalars_to_json(f"{opt.save_path}/all_scalars.json")
    writer.close()
