import logging
from ast import literal_eval as make_tuple
import torch
import numpy
import neusum
from typing import List

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 100
max_doc_len = 80
report_every = 100000
# if make shuffle = 1, we should shuffle *.ref together, or we can't track the original data
shuffle = 0
sorting = 0 # don't even know why sorting...
norm_lambda = 5

logger = logging.getLogger(__name__)


def makeVocabulary(filenames: List[str], size: int) -> neusum.Dict:
    vocab = neusum.Dict([neusum.Constants.PAD_WORD, neusum.Constants.UNK_WORD,
                         neusum.Constants.BOS_WORD, neusum.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name: str, dataFiles: List[str], vocabFile: str, vocabSize: int) -> neusum.Dict:
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name +
                    " vocabulary from '" + vocabFile + "'...")
        vocab = neusum.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def np_softmax(x, a=1):
    """Compute softmax values for each sets of scores in x."""
    return numpy.exp(a * x) / numpy.sum(numpy.exp(a * x), axis=0)


def makeData(srcFile: str, tgtFile: str, train_oracle_file: str, train_src_rouge_file: str, srcSectionFile: str, srcDicts, tgtDicts, drop_too_short: int = 10, drop_too_long: int = 500, bert_annotation: str = ''):
    """ The target files are useless in this case (tgtFile, train_src_rouge_file unused)
    (actually we don't need to load bert annotation here... we will do it in load_dev_data)
    (so just ignore all the 'if bert_annotation')
    """
    src, tgt = [], []
    srcSection = []
    src_raw = []
    srcSection_raw = []
    src_rouge = []
    oracle = []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s' % srcFile)
    srcF = open(srcFile, encoding='utf-8')
    srcSectionF = open(srcSectionFile, encoding='utf-8')
    # tgtF = open(tgtFile, encoding='utf-8')
    oracleF = open(train_oracle_file, encoding='utf-8')
    # src_rougeF = open(train_src_rouge_file, encoding='utf-8')
    if bert_annotation:
        bertF = open(bert_annotation, encoding='utf-8')
        bert_annotation_batch = []

    while True:
        sline = srcF.readline()
        secline = srcSectionF.readline()
        # tline = tgtF.readline()
        oline = oracleF.readline()
        # src_rouge_line = src_rougeF.readline()
        if bert_annotation:
            bline = bertF.readline()

        # normal end of file
        # if sline == "" and tline == "":
        if sline == "" and oline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or oline == "":
            logger.info(
                'WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        secline = secline.strip()
        # tline = tline.strip()
        oline = oline.strip()
        # src_rouge_line = src_rouge_line.strip()
        if bert_annotation:
            bline = bline.strip()

        # source and/or target are empty
        # if sline == "" or tline == "" or ('None' in oline) or ('nan' in src_rouge_line):
        # logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
        # continue

        srcSents = sline.split('##SENT##')[:max_doc_len]
        srcSectionSents = secline.split('##SENT##')[:max_doc_len]
        if len(srcSents) < drop_too_short or len(srcSents) > drop_too_long:
            logger.debug('Drop data too short or too long')
            continue

        # rouge_gains = src_rouge_line.split('\t')[1:]
        srcWords = [x.split(' ')[:seq_length] for x in srcSents]
        srcSectionWords = [x.split(' ')[:seq_length] for x in srcSectionSents]
        # tgtWords = ' '.join(tgtSents)
        # oracle_combination = make_tuple(oline.split('\t')[0])
        oracle_combination = make_tuple(oline.split('\t')[0])
        oracle_combination = [x for x in oracle_combination]  # no sentinel
        if bert_annotation:
            bert_annotation_combination = make_tuple(bline.split('\t')[0])
            bert_annotation_combination = [x for x in bert_annotation_combination]  # no sentinel

        index_out_of_range = [x >= max_doc_len for x in oracle_combination]
        # TODO here
        if any(index_out_of_range):
            logger.debug(
                'WARNING: oracle exceeds max_doc_len, ignoring (' + str(count + 1) + ')')
            continue

        src_raw.append(srcSents)
        srcSection_raw.append(srcSectionSents)

        src.append([srcDicts.convertToIdx(word,
                                          neusum.Constants.UNK_WORD) for word in srcWords])
        srcSection.append([srcDicts.convertToIdx(word,
                                          neusum.Constants.UNK_WORD) for word in srcSectionWords])
        # tgt.append(tgtWords)

        oracle.append(torch.LongTensor(oracle_combination))
        if bert_annotation:
            bert_annotation_batch.append(torch.LongTensor(bert_annotation_combination))
        # rouge_gains = [[float(gain) for gain in x.split(' ')] for x in rouge_gains]
        # rouge_gains = [numpy.array(x) for x in rouge_gains]
        # rouge_gains = [(x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) for x in rouge_gains]
        # rouge_gains = [torch.from_numpy(np_softmax(x, norm_lambda)).float() for x in rouge_gains]
        # src_rouge.append(rouge_gains)

        sizes += [len(srcWords)]

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    srcSectionF.close()
    # tgtF.close()
    oracleF.close()
    # src_rougeF.close()
    if bert_annotation:
        bertF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_raw = [src_raw[idx] for idx in perm]
        srcSection = [srcSection[idx] for idx in perm]
        srcSection_raw = [srcSection_raw[idx] for idx in perm]
        # tgt = [tgt[idx] for idx in perm]
        oracle = [oracle[idx] for idx in perm]
        # src_rouge = [src_rouge[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

        if bert_annotation:
            bert_annotation_batch = [bert_annotation_batch[idx] for idx in perm]

    if sorting == 1:
        logger.info('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        src_raw = [src_raw[idx] for idx in perm]
        srcSection = [srcSection[idx] for idx in perm]
        srcSection_raw = [srcSection_raw[idx] for idx in perm]
        # tgt = [tgt[idx] for idx in perm]
        oracle = [oracle[idx] for idx in perm]
        # src_rouge = [src_rouge[idx] for idx in perm]

        if bert_annotation:
            bert_annotation_batch = [bert_annotation_batch[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    # return src, src_raw, tgt, oracle, src_rouge

    if bert_annotation:
        return src, src_raw, None, oracle, None, srcSection, srcSection_raw, bert_annotation_batch
    else:
        return src, src_raw, None, oracle, None, srcSection, srcSection_raw, None


def prepare_data_online(train_src: str, src_vocab: str, train_tgt, tgt_vocab, train_oracle, train_src_rouge, src_section: str, drop_too_short: int = 10, drop_too_long: int = 500, bert_annotation: str = ''):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 1000000)
    # dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)
    dicts['tgt'] = None

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['src_raw'], train['tgt'], \
        train['oracle'], train['src_rouge'], \
        train['src_section'], train['src_section_raw'], train['bert_annotation'] = makeData(train_src, train_tgt,
                                                       train_oracle, train_src_rouge, src_section,
                                                       dicts['src'], dicts['tgt'],
                                                       drop_too_short, drop_too_long, bert_annotation)

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
