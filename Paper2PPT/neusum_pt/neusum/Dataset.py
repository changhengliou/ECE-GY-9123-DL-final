from __future__ import division

from typing import List

import math
import random

import torch
from torch.autograd import Variable

import neusum
import ipdb


class Dataset(object):
    def __init__(self, srcData, src_raw: List[List[str]], tgtData, oracleData, src_rouge, src_section, src_section_raw, batchSize, maxDocLen, cuda, volatile=False,
                       bert_annotation=None, good_patterns: List[str]=None, use_good: bool=False):
        """
        srcData
        src_raw: raw sentences for each paper
        tgtData: unused
        oracleData: gold label (might change by Iterative Labeling)
        src_rouge: unused
        batchSize
        maxDocLen
        cuda
        volatile
        bert_annotation: if not given, this will be the same as oracleData
        qtype: to filter good pattern data (if necessary)
        """
        self.src = srcData
        self.src_raw = src_raw
        self.src_section = src_section
        self.src_section_raw = src_section_raw
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        if oracleData:
            self.oracle = oracleData
            assert (len(self.src) == len(self.oracle))
        else:
            self.oracle = None

        if bert_annotation:
            # (For test) 
            self.bert_annotation = bert_annotation
            assert (len(self.src) == len(self.bert_annotation))
        else:
            # If not given BERT annotation, just copy one from oracleData
            # (For iterative labeling) 
            self.bert_annotation = self.oracle
        
        # Mark the good quality data, mask other while calculating loss
        self.use_good = use_good
        if good_patterns:
            self.good_patterns = good_patterns
            self._init_good_paper()
        else:
            self.good_paper = None

        if src_rouge:
            self.src_rouge = src_rouge
            assert (len(self.src) == len(self.src_rouge))
        else:
            self.src_rouge = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.maxDocLen = maxDocLen
        self.numBatches = math.ceil(len(self.src) / batchSize)
        self.volatile = volatile
    
    def _init_good_paper(self):
        """ Same logic as label_select.py, only include the paper which contain certain pattern """
        self.good_paper = [False] * len(self.src_raw)
        for i in range(len(self.src_raw)):
            # if any(pattern in ' '.join(self.src_raw[i]) for pattern in self.good_patterns):
            if any(pattern.lower() in ' '.join(self.src_raw[i]).lower() for pattern in self.good_patterns):
                self.good_paper[i] = True


    def _batchify(self, data, align_right=False, include_lengths=False):
        """ padding to max length """
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(neusum.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        batch_src_data = self.src[index * self.batchSize:(index + 1) * self.batchSize]
        src_raw = self.src_raw[index * self.batchSize:(index + 1) * self.batchSize]
        doc_lengths = []
        buf = []

        batch_src_section_data = self.src_section[index * self.batchSize:(index + 1) * self.batchSize]
        src_section_raw = self.src_section_raw[index * self.batchSize:(index + 1) * self.batchSize]
        section_doc_lengths = []
        section_buf = []

        for item in batch_src_data:
            doc_lengths.append(min(len(item), self.maxDocLen))
            buf += item[:self.maxDocLen]
            if len(item) < self.maxDocLen:
                buf += [torch.LongTensor([neusum.Constants.PAD]) for _ in range(self.maxDocLen - len(item))]
        for item in batch_src_section_data:
            section_doc_lengths.append(min(len(item), self.maxDocLen))
            section_buf += item[:self.maxDocLen]
            if len(item) < self.maxDocLen:
                section_buf += [torch.LongTensor([neusum.Constants.PAD]) for _ in range(self.maxDocLen - len(item))]

        srcBatch, lengths = self._batchify(buf, align_right=False, include_lengths=True)
        srcSectionBatch, sectionLengths = self._batchify(section_buf, align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self.tgt[index * self.batchSize:(index + 1) * self.batchSize]
        else:
            tgtBatch = None
        if self.oracle:
            oracleBatch, oracleLength = self._batchify(
                self.oracle[index * self.batchSize:(index + 1) * self.batchSize],
                include_lengths=True)
        else:
            oracleBatch = None
        if self.bert_annotation:
            bertBatch, bertLength = self._batchify(
                self.bert_annotation[index * self.batchSize:(index + 1) * self.batchSize],
                include_lengths=True)
        else:
            bertBatch = None
        if self.src_rouge:
            buf = []
            max_points = max(oracleLength)
            batch_src_rouge_gain_data = self.src_rouge[index * self.batchSize:(index + 1) * self.batchSize]
            for item in batch_src_rouge_gain_data:
                buf += [x[:self.maxDocLen] for x in item]
                if len(item) < max_points:
                    buf += [torch.FloatTensor([neusum.Constants.PAD]) for _ in range(max_points - len(item))]
            src_rouge_batch = self._batchify(buf)
        else:
            src_rouge_batch = None

        # TODO: I don't think we need to sort here..., but srcBatch need to be a tuple?!
        # within batch sorting by decreasing length for variable length rnns
        # TODO: src_raw & src_section_raw didn't sort, not sure if there will be any problem
        # import ipdb; ipdb.set_trace()
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch, srcSectionBatch, sectionLengths)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, srcBatch, srcSectionBatch, sectionLengths = zip(*batch)
        # import ipdb; ipdb.set_trace()

        def wrap(b):
            """ this contain a transpose """
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            # b = Variable(b, volatile=self.volatile)
            b = Variable(b)
            return b

        def simple_wrap(b):
            if b is None:
                return b
            if self.cuda:
                b = b.cuda()
            # b = Variable(b, volatile=self.volatile)
            b = Variable(b)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        # lengths = Variable(lengths, volatile=self.volatile)
        lengths = Variable(lengths)

        doc_lengths = torch.LongTensor(doc_lengths).view(1, -1)
        # doc_lengths = Variable(doc_lengths, volatile=self.volatile)
        doc_lengths = Variable(doc_lengths)

        sectionLengths = torch.LongTensor(sectionLengths).view(1, -1)
        sectionLengths = Variable(sectionLengths)
        section_doc_lengths = torch.LongTensor(section_doc_lengths).view(1, -1)
        section_doc_lengths = Variable(section_doc_lengths)


        if self.oracle:
            oracleLength = torch.LongTensor(oracleLength).view(1, -1)
            # oracleLength = Variable(oracleLength, volatile=self.volatile)
            oracleLength = Variable(oracleLength)
        else:
            oracleLength = None

        if self.bert_annotation:
            bertLength = torch.LongTensor(bertLength).view(1, -1)
            bertLength = Variable(bertLength)
        else:
            bertLength = None

        if self.use_good and self.good_paper:
            # TODO: maybe directly return good_paper_marker_batch?! (with bad indices) and calculate it in train (random with prob increase...)
            good_paper_marker_batch = self.good_paper[index * self.batchSize:(index + 1) * self.batchSize]
            bad_indices = [i for i, val in enumerate(good_paper_marker_batch) if not val]
        else:
            # Assume all paper are good
            good_paper_marker_batch = [True] * self.batchSize
            bad_indices = []


        # return (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
        #        (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
        #        simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)
        
        # batch[0]
        # batch[1]
        # batch[2]
        # batch[3]
        # batch[4]
        # batch[5]: good bad paper
        # batch[6]: section info, tensors
        # batch[7]: section raw text
        # batch[8]: start, end of this batch (used to update data)
        return (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), \
               (simple_wrap(bertBatch), bertLength), \
               (good_paper_marker_batch, bad_indices), \
               (wrap(srcSectionBatch), sectionLengths, section_doc_lengths), (src_section_raw,), \
               (index * self.batchSize, min((index + 1) * self.batchSize, len(self.src)))

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.src_raw, self.oracle, self.src_section, self.src_section_raw, self.bert_annotation))
        self.src, self.src_raw, self.oracle, self.src_section, self.src_section_raw, self.bert_annotation, = zip(
            *[data[i] for i in torch.randperm(len(data))])
