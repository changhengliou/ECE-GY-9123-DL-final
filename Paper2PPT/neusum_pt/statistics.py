import loglinear
import neusum
import onlinePreprocess
from onlinePreprocess import prepare_data_online
import argparse
import xargs
# https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778
from torchsummary import summary
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='statistics')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()


def get_n_params(model):
    """ https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6 """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_trainable_params(model):
    """ https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6 """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    return params

def get_report(model: nn.Module, model_name: str):
    print('Parameters of the', model_name)
    print(get_n_params(model))
    print('Trainable parameters of the', model_name)
    print(get_trainable_params(model))
    # print('Torchsummary report of the', model_name)
    # summary(model)



def main():
    onlinePreprocess.seq_length = opt.max_sent_length
    onlinePreprocess.max_doc_len = opt.max_doc_len
    onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
    onlinePreprocess.norm_lambda = opt.norm_lambda
    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                    opt.train_src_rouge, opt.train_src_section, opt.drop_too_short, opt.drop_too_long)

    trainData = neusum.Dataset(dataset['train']['src'], dataset['train']['src_raw'], dataset['train']['tgt'],
                               dataset['train']['oracle'], dataset['train']['src_rouge'], dataset[
                                   'train']['src_section'], dataset['train']['src_section_raw'],
                               opt.batch_size, opt.max_doc_len, opt.gpus, dataset[
                                   'train']['bert_annotation'],
                               good_patterns=loglinear.Config.Keyword[opt.qtype], use_good=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d' %
                (dicts['src'].size()))
    print(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

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

    log_linear_model = loglinear.model.LogLinear()
    log_linear_model.set_rules(opt.position_weight, opt.keyword_weight,
                                loglinear.Config.Keyword[opt.qtype], opt.in_bert_weight,
                                opt.in_section_weight, loglinear.Config.PossibleSection[opt.qtype],
                                opt.section_embedding, opt.pre_word_vecs_enc)

    get_report(model, 'Neural-based Model')
    get_report(log_linear_model, 'Log-linear Model')

if __name__ == "__main__":
    main()