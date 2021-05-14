import onlinePreprocess
import xargs
import neusum
import torch
import torch.nn as nn
import os
from glob import glob
import argparse
import logging
import time
import math

parser = argparse.ArgumentParser(description='test.py')
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
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8', mode='a')
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

if __name__ == "__main__":
    # put this here so it will use the logger of test
    from train import evalModel, load_dev_data

    if opt.online_process_data:
        logger.info('Online Preprocessing data (to get vocabulary dictionary).')
        onlinePreprocess.seq_length = opt.max_sent_length
        onlinePreprocess.max_doc_len = opt.max_doc_len
        onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
        onlinePreprocess.norm_lambda = opt.norm_lambda
        from onlinePreprocess import prepare_data_online
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                    opt.train_src_rouge, opt.train_src_section, opt.drop_too_short, opt.drop_too_long)
    else:
        logger.info('Use preprocessed data stored in checkpoint.')
        dataset = {} # this is used for the summarizer (only need the 'dict' part)

    logger.info('Loading checkpoint...')
    if opt.specific_epoch > 0:
        model_selected = os.path.join(
            opt.save_path, 'model_epoch_%s.pt' % opt.specific_epoch)
        logger.info('Loading from the specific epoch checkpoint "%s"' %
                    model_selected)
    else:
        # Find the latest model to load
        model_path = glob(os.path.join(opt.save_path, '*.pt'))
        if not model_path:
            raise ValueError("Can't find model %s" %
                             os.path.join(opt.save_path, '*.pt'))

        # make sure not load the log linear model
        model_selected = None
        for candidate in reversed(sorted(model_path, key=os.path.getmtime)):
            if 'log_linear' not in candidate:
                model_selected = candidate
                break
        assert model_selected is not None
        logger.info('Loading from the latest model "%s"' % model_selected)

    if opt.gpus:
        checkpoint = torch.load(model_selected)
    else:
        checkpoint = torch.load(
            model_selected, map_location=torch.device('cpu'))

    logger.info('\tprevious training epochs: %d' % checkpoint['epoch'])

    if not opt.online_process_data:
        dataset['dicts'] = checkpoint['dicts']
    dicts = checkpoint['dicts']
    # ==============

    # logger.info(' * vocabulary size. source = %d; target = %d' %
    #             (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * vocabulary size. source = %d' %
                (dicts['src'].size()))
    if opt.online_process_data:
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

    # load model
    logger.info('Loading trained model...')
    # model.load_state_dict(checkpoint['model'])

    model.load_state_dict(checkpoint['model'])
    summarizer = neusum.Summarizer(opt, model, dataset)

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    testData = load_dev_data(summarizer, opt.dev_input_src, opt.dev_ref,
                             opt.dev_input_src_section, test_bert_annotation=opt.test_bert_annotation)
    model.eval()
    scores = evalModel(model, summarizer, testData,
                       opt.output_len, 'test', opt.set_postfix, opt.stripping_mode, checkpoint['epoch'])
    logger.info('Using checkpoint: %s' % model_selected)
    logger.info('Key hyperparmeters:')
    # the setting is not consistent with the train
    # logger.info('\tMax Doc Length: %s' % opt.max_doc_len)
    # logger.info('\tForce max length at test %s' % opt.force_max_len)
    # logger.info('\tIterative Labeling start epoch: %s' % opt.relabel_epoch)
    logger.info('\tMax Decode Steps: %d' % opt.max_decode_step)
    logger.info('\tKeep Data: [%d, %d)' %
                (opt.drop_too_short, opt.drop_too_long))
    logger.info('\tTrained epoch: %s' % checkpoint['epoch'])
    logger.info(
        'Evaluate score: (accuracy) total / hit@1, {precision, recall, f1 score} (sentence-level)')
    logger.info(scores)
