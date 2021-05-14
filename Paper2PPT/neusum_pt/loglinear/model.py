# import sys
# sys.path.append('/mnt/d/Program/BERTIterativeLabeling/Paper2PPT/neusum_pt')
import torch
from neusum.Models import Encoder
import neusum.Dataset
import torch.nn as nn
from typing import List
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class SentEncoder(Encoder):
    """ Unused
    (careful if want to use this.
    currently the srcBatch were sorted in dataset,
    but the src_raw didn't,
    so might have mismatch with other features)
    """

    def __init__(self, opt, dicts):
        super(SentEncoder, self).__init__(opt, dicts)

    def forward(self, x):
        """
        x: (wrap(srcBatch), lengths)
        """
        lengths = input[1].data.view(-1).tolist(
        )  # lengths data is wrapped inside a Variable
        if self.freeze_word_vecs_enc:
            wordEmb = self.word_lut(input[0]).detach()
        else:
            wordEmb = self.word_lut(input[0])


class LogLinear(nn.Module):
    def __init__(self, sent_encoder: SentEncoder = None, use_gpu: bool = False, add_one: bool = True, is_non_linear: bool = False):
        super(LogLinear, self).__init__()
        self.sent_encoder = sent_encoder
        self.use_gpu = use_gpu
        self.add_one = add_one
        self.is_non_linear = is_non_linear
        self.rules_used = []
        self.dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.gamma = None

        self.linear = None
        self.linear2 = None
        self.non_linear_hidden_layer_dim = 32
        if self.is_non_linear:
            self.linear = nn.Linear(1, self.non_linear_hidden_layer_dim)
            self.linear2 = nn.Linear(self.non_linear_hidden_layer_dim, 1)
            if self.use_gpu:
                self.linear.to('cuda')
                self.linear2.to('cuda')

    def gen_mask_with_length(self, doc_len, batch_size, lengths):
        """ copied from DocumentEncoder """
        if self.use_gpu:
            mask = torch.ByteTensor(batch_size, doc_len).cuda().zero_()
        else:
            mask = torch.ByteTensor(batch_size, doc_len).zero_()

        ll = lengths.data.view(-1).tolist()
        for i in range(batch_size):
            for j in range(doc_len):
                if j >= ll[i]:
                    mask[i][j] = 1
        mask = mask.float()
        return mask

    def set_rules(self, position_in_the_whole_paper: float = -1.0,
                  contain_keywords: float = -1.0, key_words_list: List[str] = [],
                  in_bert_prediction: float = -1.0,
                  in_section: float = -1.0, section_list: List[str] = [],
                  use_section_embedding: bool = False, path_to_glove: str = ''):
        """ Given weight to each rule (if set negative values then it will disable them) """
        weights = []
        dimension = 0
        if position_in_the_whole_paper >= 0:
            self.rules_used.append('position_in_the_whole_paper')
            dimension += 1
            weights.append(position_in_the_whole_paper)

        if contain_keywords >= 0:
            self.rules_used.append('contain_keywords')
            dimension += 1
            weights.append(contain_keywords)
            self.key_words_list = key_words_list

        if in_bert_prediction >= 0:
            self.rules_used.append('in_bert_prediction')
            dimension += 1
            weights.append(in_bert_prediction)

        if in_section >= 0:
            self.rules_used.append('in_section')
            dimension += 1
            weights.append(in_section)
            self.section_list = section_list        

        if use_section_embedding and path_to_glove:
            self.rules_used.append('section_embedding')
            tmp_file = get_tmpfile('word2vec.txt')
            num_word, num_dim = glove2word2vec(path_to_glove, tmp_file)
            self.w2v_model = KeyedVectors.load_word2vec_format(tmp_file)
            dimension += num_dim
            section_embedding = [1] * num_dim
            self.embedding_dim = num_dim
            weights.extend(section_embedding)

        weights = self.dtype(weights).view(1, len(weights))
        with torch.no_grad():
            # https://discuss.pytorch.org/t/how-to-initialize-weight-with-arbitrary-tensor/3432/2
            # (deprecated) https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/3
            # https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch
            self.gamma = nn.Linear(
                weights.shape[0], weights.shape[1], bias=False)
            self.gamma.weight = nn.Parameter(weights)


    def get_position_in_the_whole_paper(self, batch):
        """ set position index start from 1. for the length exceeded set to 0
        (normalize with the max length of current batch)
        """
        length = batch[0][2]
        max_length = torch.max(length)

        tensors = []
        for l in length.view(-1):
            temp = torch.arange(1, max_length+1).type(self.dtype)
            temp[l:] = 0.0
            temp /= max_length
            tensors.append(temp)
        return torch.stack(tensors)

    def get_keyword_feature(self, batch, accumulate: bool = True, amplification: float = 1, case_sensitive: bool = False):
        """ if a sentence contain any keyword then set to 1
        (if multiple match, maybe we can accumulate the value then normalize it?!)
        
        accumulate: if match multiple keyword, it will accumulate the scores
        amplification: default match one will gain 1 score, otherwise amplify the score with this constant
        """
        sents_in_docs = batch[1][0]
        max_length = torch.max(batch[0][2])

        tensor = torch.zeros((len(sents_in_docs), max_length)).type(self.dtype)
        max_count = self.dtype([0.0])

        for i, doc in enumerate(sents_in_docs):
            for j, sent in enumerate(doc):
                for keyword in self.key_words_list:
                    if not case_sensitive:
                        keyword = keyword.lower()
                        sent = sent.lower()
                            
                    if accumulate:
                        if keyword in sent:
                            tensor[i, j] += 1.0 * amplification
                            max_count = torch.max(max_count, tensor[i, j])
                    else:
                        # 1 if contain any keyword
                        if keyword in sent:
                            tensor[i, j] = 1.0 * amplification
                            break

        if accumulate and max_count > 0:
                tensor /= max_count

        return tensor

    def get_oracle_targets(self, batch):
        """ if the bert annotate the sentence then set to 1.
        (This might have problem when use test data) => Deprecated
        """
        oracle_targets = batch[2][0]
        oracle_lengths = batch[2][1]
        max_length = torch.max(batch[0][2])
        tensor = torch.zeros(
            oracle_targets.shape[0], max_length).type(self.dtype)
        for i, l in enumerate(oracle_lengths.view(-1)):
            for idx in oracle_targets[i][:l]:
                tensor[i, idx] = 1.0

        return tensor

    def get_bert_annotation(self, batch):
        """ if the bert annotate the sentence then set to 1.
        """
        oracle_targets = batch[4][0]
        oracle_lengths = batch[4][1]
        max_length = torch.max(batch[0][2])
        tensor = torch.zeros(
            oracle_targets.shape[0], max_length).type(self.dtype)
        for i, l in enumerate(oracle_lengths.view(-1)):
            for idx in oracle_targets[i][:l]:
                tensor[i, idx] = 1.0
        
        return tensor
    
    def get_probabily_in_section(self, batch, amplification: float = 1):
        """ if this question type is probabily in this section, then set to 1
        """
        section_of_sents = batch[7][0]
        max_length = torch.max(batch[0][2])

        tensor = torch.zeros((len(section_of_sents), max_length)).type(self.dtype)

        for i, doc in enumerate(section_of_sents):
            for j, section in enumerate(doc):
                for keyword in self.section_list:
                    # 1 if contain any keyword
                    if keyword.lower() in section.lower():
                        tensor[i, j] = 1.0 * amplification
                        break

        return tensor

    def get_section_embedding(self, batch):
        """ get embedding of the section
        """
        section_of_sents = batch[7][0]
        section_name_length = [len(name) for name in section_of_sents]
        max_length = max(section_name_length)

        tensor = torch.zeros(len(section_name_length), max_length, self.embedding_dim).type(self.dtype)
        for i, l in enumerate(section_name_length):
            cnt = 0
            for j in range(l):
                try:
                    tensor[i, j, :] += torch.tensor(self.w2v_model[section_of_sents[i][j].lower()]).type(self.dtype)
                    cnt += 1
                except KeyError:
                    # UNK
                    pass
            
            # tensor[i, :, :] /= l
            if cnt > 0:
                tensor[i, :, :] /= cnt
        
        return tensor
            

    def add_one_for_all_feature(self, batch, feature_tensors):
        """
        Make every sentence starts with score = 1, as to differentiate
        extra padding sentences (score=0) in the softmax in regression_loss()
        """
        length = batch[0][2][0]
        for b, l in enumerate(length):
            feature_tensors[b, :l, :] += 1.0


    def forward(self, batch: neusum.Dataset):
        """
        Input will be a batch from trainData object

        batch: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)
        """
        assert self.gamma is not None  # make sure you've set the rules

        # if self.sent_encoder:
        #     enc_hidden, context = self.sent_encoder(batch[0])

        feature_tensors = []
        if 'position_in_the_whole_paper' in self.rules_used:
            feature_tensors.append(self.get_position_in_the_whole_paper(batch))

        if 'contain_keywords' in self.rules_used:
            feature_tensors.append(self.get_keyword_feature(batch))

        if 'in_bert_prediction' in self.rules_used:
            feature_tensors.append(self.get_bert_annotation(batch))
        
        if 'in_section' in self.rules_used:
            feature_tensors.append(self.get_probabily_in_section(batch))
        
        # (self.get_bert_annotation(batch).max(dim=1) and self.get_bert_annotation(batch).max(dim=1) should be close in train)

        # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-long-for-sequence-element-1-in-sequence-argument-at-position-1-tensors/46952/2
        feature_tensors = torch.stack(feature_tensors, dim=2)

        if self.add_one:
            self.add_one_for_all_feature(batch, feature_tensors)

        if 'section_embedding' in self.rules_used:
            feature_tensors = torch.cat((feature_tensors, self.get_section_embedding(batch)), dim=2)
    
        # (batch, max_doc_length, n_feature) * (n_feaure, 1)
        scores = self.gamma(feature_tensors)
        # (batch, max_doc_length, 1)

        if self.is_non_linear:
            scores = torch.sigmoid(scores)
            scores = self.linear(scores)
            # (batch, max_doc_length, self.non_linear_hidden_layer_dim)
            scores = torch.sigmoid(scores)
            scores = self.linear2(scores)
            # (batch, max_doc_length, 1)

        scores = scores.squeeze(-1)
        # (batch, max_doc_length)

        doc_sent_mask = self.gen_mask_with_length(
            torch.max(batch[0][2]), len(batch[1][0]), batch[0][2])
        doc_sent_mask = torch.autograd.Variable(
            doc_sent_mask, requires_grad=False)
        return scores, doc_sent_mask


if __name__ == "__main__":

    # test the modules
    import onlinePreprocess
    onlinePreprocess.seq_length = 80
    onlinePreprocess.max_doc_len = 500
    onlinePreprocess.shuffle = 0
    onlinePreprocess.norm_lambda = 20
    from onlinePreprocess import prepare_data_online
    dataset = prepare_data_online("../../data/train/future/train.txt.src", None, None, None, "../../data/train/future/train.txt.oracle",
                                  None, "../../data/train/future/train.txt.section",
                                  10, 500, '')

    trainData = neusum.Dataset(dataset['train']['src'], dataset['train']['src_raw'],
                               dataset['train']['tgt'], dataset['train']['oracle'], dataset['train']['src_rouge'],
                               dataset['train']['src_section'], dataset['train']['src_section_raw'], 4, 500, None)

    model = LogLinear()
    # model.set_rules(1.0, 1.0, ['future'], 1.0)
    from loglinear.Config import Keyword, PossibleSection

    # use section title average embedding
    # model.set_rules(1.0, 1.0, Keyword['future'], 1.0, -1.0, [], True, '../glove/glove.6B.50d.txt')

    # use binary one
    model.set_rules(1.0, 1.0, Keyword['future'], 1.0, 1.0, PossibleSection['future'], False, '')

    for i in range(len(trainData)):
        batch = trainData[i]
        doc_sent_scores, doc_sent_mask = model(batch)
