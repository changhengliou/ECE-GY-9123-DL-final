from typing import List
import neusum
import torch.nn as nn
import torch
from torch.autograd import Variable

try:
    import ipdb
except ImportError:
    pass


# Important notes:
#
# Based on tracing code,
# the "scores" of the Beam is the score of each "sentence order" not "prediction score" of each step
# The "prediction score" is the attention score, at each step it will mask the last pick sentence to be -inf

class Summarizer(object):
    def __init__(self, opt, model:neusum.Models.NMTModel=None, dataset=None, logger=None):
        self.opt = opt

        if model is None: # currently, when we use summarizer we will always input model, so won't execute this
            checkpoint = torch.load(opt.model)

            model_opt = checkpoint['opt']
            if logger is not None:
                logger.info('Loading model from {0}'.format(opt.model))
                logger.info('model_opt')
                logger.info(model_opt)
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.enc_rnn_size = model_opt.doc_enc_size
            self.dec_rnn_size = model_opt.dec_rnn_size
            sent_encoder = neusum.Models.Encoder(model_opt, self.src_dict)
            doc_encoder = neusum.Models.DocumentEncoder(model_opt)
            pointer = neusum.Models.Pointer(model_opt, self.tgt_dict)
            if hasattr(model_opt, 'dec_init'):
                if model_opt.dec_init == "simple":
                    decIniter = neusum.Models.DecInit(model_opt)
                elif model_opt.dec_init == "att":
                    decIniter = neusum.Models.DecInitAtt(model_opt)
                else:
                    raise ValueError('Unknown decoder init method: {0}'.format(model_opt.dec_init))
            else:
                # TODO: some old model do not have this attribute in it
                decIniter = neusum.Models.DecInit(model_opt)

            model = neusum.Models.NMTModel(sent_encoder, doc_encoder, pointer, decIniter, None)

            model.load_state_dict(checkpoint['model'])

            if opt.cuda:
                model.cuda()
            else:
                model.cpu()

        else:
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']

            self.enc_rnn_size = opt.doc_enc_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            # self.opt.n_best = 1 # It just hard coded the n_best
            # self.opt.n_best = 5 # For debug purpose
            self.opt.n_best = opt.n_best_size # better to just set 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, srcRaw, tgtRaw, oracleBatch, srcRougeBatch, src_section_batch, src_section_raw, bert_annotation=None, good_patterns: List[str] = None, use_good: bool = False):
        """
        (used in load_dev_data)
        """
        srcData = [[self.src_dict.convertToIdx(b,
                                               neusum.Constants.UNK_WORD) for b in doc] for doc in srcBatch]
        srcBatchData = [[self.src_dict.convertToIdx(b,
                                               neusum.Constants.UNK_WORD) for b in doc] for doc in src_section_batch]

        return neusum.Dataset(srcData, srcRaw, tgtRaw, oracleBatch, srcRougeBatch, srcBatchData, src_section_raw, self.opt.batch_size,
                           # self.opt.max_doc_len, self.opt.cuda, volatile=True)
                           self.opt.max_doc_len, self.opt.cuda, bert_annotation=bert_annotation, good_patterns=good_patterns, use_good=use_good)

    def buildTargetTokens(self, pred, src, attn):
        tokens = self.tgt_dict.convertToLabels(pred, neusum.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == neusum.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, batch):
        """
        input: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (oracleBatch, oracleLength), \
               Variable(torch.LongTensor(list(indices)).view(-1).cuda(), volatile=self.volatile)
        return:
              allHyp: the predict "label" (length equal to max_decode_step)
              allScores: the score of the "sentence order" of each prediction collection (length equal to n_best)
              allAttn: the attention score of each step in beam search of the specific sentence order (shape equal to [max_decode_step, attention size])
        """

        batchSize = batch[0][2].size(1)
        beamSize = self.opt.beam_size

        #  (1) Encode the document
        srcBatch = batch[0]
        # indices = batch[4]
        indices = batch[3]
        doc_hidden, doc_context, doc_sent_mask = self.model.encode_document(srcBatch, indices)

        if isinstance(self.model.decIniter, neusum.Models.DecInitAtt):
            enc_hidden = self.model.decIniter(doc_context, doc_sent_mask)
        elif isinstance(self.model.decIniter, neusum.Models.DecInit):
            if self.model.decIniter.num_directions == 2:
                enc_hidden = self.model.decIniter(doc_hidden[1])  # [1] is the last backward hiden
            else:
                enc_hidden = self.model.decIniter(doc_hidden[0])
        else:
            raise ValueError("Unknown decIniter type")

        pointer_precompute, reg_precompute = None, None

        decStates = enc_hidden  # batch, dec_hidden

        # Expand tensors for each beam.
        context = Variable(doc_context.data.repeat(1, beamSize, 1))
        decStates = Variable(decStates.unsqueeze(0).data.repeat(1, beamSize, 1))
        # pointer_att_vec = self.model.make_init_att(context)
        reg_att_vec = self.model.make_init_att(context)
        # padMask = Variable(doc_sent_mask.data.repeat(beamSize, 1), volatile=True)
        padMask = Variable(doc_sent_mask.data.repeat(beamSize, 1))
        baseMask = doc_sent_mask.data.clone()

        # Create Beam class for each data of a batch ("batchSize" sentences)
        beam = [neusum.Beam(beamSize, self.opt.cuda, self.opt.force_max_len) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        # Decoding with beam search
        for i in range(self.opt.max_decode_step):
            # Prepare decoder input. (get the outputs for the current timestep)
            # (according to the code of Beam.py, it seems b.done will always be False)
            # this "input" variable is never used now
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            if i > 0:
                all_masks = torch.stack(
                    [b.get_doc_sent_mask(baseMask[idx]) for idx, b in enumerate(beam)
                     if not b.done]).transpose(0, 1).contiguous()
                # all_masks = Variable(all_masks.view(-1, all_masks.size(2)), volatile=True)
                all_masks = Variable(all_masks.view(-1, all_masks.size(2)))
            else:
                all_masks = padMask
            decStates, attn, reg_att_vec, reg_precompute = self.model.pointer(
                decStates, context, padMask, [all_masks],
                reg_att_vec, reg_precompute,
                1, reg_att_vec)

            # batch x beam x numWords
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            # Find active sentences again
            active = [] # store the batch index
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    # Skip all the sentences which are finished decoding
                    # (but seens currently no sentence will be set to done)
                    continue

                idx = batchIdx[b]
                # currently the value for each batch (i.e. attn.data[idx]), the value are all the same
                if not beam[b].advance(attn.data[idx]):
                    active += [b] # since beam[b].advance will never done (i.e. return True), thus all batches will always be active
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            # (but since the sentences will never completed since beam[b].done is always false)
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam_idx: idx for idx, beam_idx in enumerate(active)}

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                # .view(*newSize), volatile=True)
                                .view(*newSize))

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            reg_att_vec = updateActive(reg_att_vec, self.enc_rnn_size)
            reg_precompute = None
            padMask = updateActive(padMask, padMask.size(1))

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            # previous_index = Variable(previous_index, volatile=True)
            previous_index = Variable(previous_index)
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            reg_att_vec = reg_att_vec.view(-1, reg_att_vec.size(1)).index_select(0, previous_index.view(
                -1)).view(*reg_att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            # not sure why zip, but this will make hyps and attn in the tuple of itself
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            allAttn.append(attn)

        return allHyp, allScores, allAttn

    def translate(self, src_batch, src_raw, tgt_raw):
        #  (1) convert words to indexes
        dataset = self.buildData(src_batch, src_raw, tgt_raw, None, None)
        # (wrap(srcBatch),  lengths), (wrap(tgtBatch), ), indices
        batch = dataset[0]

        #  (2) translate
        pred, predScore, attn = self.translateBatch(batch)

        #  (3) convert indexes to words
        predBatch = []
        predict_id = []
        src_raw = batch[1][0]
        for b in range(len(src_raw)):
            predBatch_nbest = []
            predict_id_nbest = []
            for n in range(self.opt.n_best):
                n = 0
                selected_sents = []
                selected_id = []
                for idx in pred[b][n]:
                    if idx >= len(src_raw[b]):
                        break
                    selected_sents.append(src_raw[b][idx])
                    selected_id.append(idx)
                predBatch_nbest.append(' '.join(selected_sents))
                predict_id_nbest.append(tuple(selected_id))
            predBatch.append(predBatch_nbest)
            predict_id.append(predict_id_nbest)

        # None seems to preserve the position for gold score?!
        return predBatch, predict_id, predScore, None
