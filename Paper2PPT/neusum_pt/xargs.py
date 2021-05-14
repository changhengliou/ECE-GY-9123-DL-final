import argparse

try:
    import ipdb
except ImportError:
    pass


def add_data_options(parser):
    # Data options
    parser.add_argument('-save_path', default='',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")

    # tmp solution for load issue
    parser.add_argument('-online_process_data', action="store_true")
    parser.add_argument('-process_shuffle', action="store_true")
    parser.add_argument('-train_src')
    parser.add_argument('-train_src_section')
    parser.add_argument('-src_vocab')
    parser.add_argument('-train_tgt')
    parser.add_argument('-tgt_vocab')
    parser.add_argument('-train_oracle')
    parser.add_argument('-train_src_rouge')
    # should be large enough to ensure all the sentences are in the input
    parser.add_argument('-max_doc_len', type=int, default=80)

    parser.add_argument('-drop_too_short', type=int, default=10)
    parser.add_argument('-drop_too_long', type=int, default=500)

    # Test options
    parser.add_argument('-dev_input_src', type=str,
                        help='Path to the dev input file.')
    parser.add_argument('-dev_input_src_section', type=str,
                        help='Path to the dev section input file.')
    parser.add_argument('-dev_ref', type=str,
                        help='Path to the dev reference file.')
    parser.add_argument('-beam_size', type=int, default=12,
                        help='Beam size')
    parser.add_argument('-max_sent_length', type=int, default=100,
                        help='Maximum sentence length.')
    parser.add_argument('-max_decode_step', type=int, default=3,
                        help='Maximum sentence length.')
    parser.add_argument('-force_max_len', action="store_true")
    parser.add_argument('-n_best_size', default=1, type=int,  # deprecated
                        help='To set the n_best (get n best sentence order score).')
    parser.add_argument('-output_len', default=5, type=int,
                        help='To output how many item for each time step when generating the *.out.* files.')
    parser.add_argument('-set_postfix', default='', type=str,
                        help='Just add the postfix in the output filename.')
    parser.add_argument('-specific_epoch', default=-1, type=int,
                        help='Load a specific epoch checkpoint')

    parser.add_argument('-eval_test_during_train', action='store_true',
                        help='Eval model during training')
    parser.add_argument('-test_input_src', type=str,
                        help='Path to the test input file.')
    parser.add_argument('-test_input_src_section', type=str,
                        help='Path to the test section input file.')
    parser.add_argument('-test_ref', type=str,
                        help='Path to the test reference file.')
    
    parser.add_argument('-test_bert_annotation', default='', type=str,
                        help='Path to the test bert annotation input file.')

    parser.add_argument('-stop_em_and_loglinear_after', default=9999, type=int,
                        help='Stop EM (teacher model i.e. log-linear model) and back to normal training or iterative labeling with NLL loss.')

def add_model_options(parser):
    # Model options
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-sent_enc_size', type=int, default=256,
                        help='Size of LSTM hidden states')
    parser.add_argument('-doc_enc_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-dec_rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=300,
                        help='Word embedding sizes')
    parser.add_argument('-att_vec_size', type=int, default=512,
                        help='Concat attention vector sizes')
    parser.add_argument('-maxout_pool_size', type=int, default=2,
                        help='Pooling size for MaxOut layer.')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    # parser.add_argument('-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")
    parser.add_argument('-sent_brnn', action='store_true',
                        help='Use a bidirectional sentence encoder')
    parser.add_argument('-doc_brnn', action='store_true',
                        help='Use a bidirectional document encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")
    parser.add_argument('-use_self_att', action='store_true',
                        help='Use self attention in document encoder.')
    parser.add_argument('-self_att_size', type=int, default=512)
    parser.add_argument('-norm_lambda', type=int,
                        help='The scale factor for normalizing the ROUGE regression score. exp(ax)/sum(exp(ax))')
    parser.add_argument('-dec_init', required=True,
                        help='Sentence encoder type: simple | att')
    # output
    parser.add_argument('-stripping_mode', type=str, default='none',
                        choices=['none', 'normal', 'diff', 'magnitude'],
                        help='The stripping mode for the output')
    parser.add_argument('-threshold', type=float, default=0.0001,
                        help='The threshold of stripping output (only work when stripping mode is normal or magnitude).')
    
    # log-linear model
    parser.add_argument('-enable_log_linear', action='store_true',
                        help='Use log-linear model during training.')
    parser.add_argument('-start_train_with_em', type=int, default=30,
                        help='The epoch to start update the teacher model with student model interactively')
    parser.add_argument('-position_weight', type=float, default=-1.0, # negative value means disable
                        help='Weight for the rule of the position in the whole paper.')
    parser.add_argument('-keyword_weight', type=float, default=-1.0, # negative value means disable
                        help='Weight for the rule of containing keyword in the sentences.')
    parser.add_argument('-in_bert_weight', type=float, default=-1.0, # negative value means disable
                        help='Weight for the rule of the label in BERT annotation.')
    parser.add_argument('-in_section_weight', type=float, default=-1.0, # negative value means disable
                        help='Weight for the rule of the keyword in section titles.')
    parser.add_argument('-section_embedding', action='store_true',
                        help='If active section embedding as one of the features.')
    parser.add_argument('-log_linear_save_path', type=str, default='',
                        help='Path to the log linear checkpoints.')
    parser.add_argument('-qtype', type=str, default='.',
                        help='To specify the keywords of which question type.')
    parser.add_argument('-IL_with_KLDivLoss', action='store_true',
                        help='Use KLDivLoss with previous epoch distribution.')
    parser.add_argument('-relabel_once', action='store_true',
                        help='Used to disable iterative labeling and use the last prediction of EM as label')
    parser.add_argument('-loglinear_learning_rate', type=float, default=0.01, 
                        help="The log-linear model's learning rate. (if we use log-linear model)")


def add_train_options(parser):
    # Optimization options
    parser.add_argument('-relabel_epoch', type=int, default=3,
                        help='Min epoch start iterative labeling')
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-optim', default='sgd',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-max_weight_value', type=float, default=15,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-sent_dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-doc_dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-dec_dropout', type=float, default=0,
                        help='Dropout probability; applied between LSTM stacks.')

    # training shuffle
    parser.add_argument('-curriculum', type=int, default=1,
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")

    # learning rate
    parser.add_argument('-learning_rate', type=float, default=1.0,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    parser.add_argument('-start_eval_batch', type=int, default=15000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-eval_per_batch', type=int, default=1000,
                        help="""evaluate on dev per x batches.""")
    parser.add_argument('-halve_lr_bad_count', type=int, default=6,
                        help="""evaluate on dev per x batches.""")

    # pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-freeze_word_vecs_enc', action="store_true",
                        help="""Update encoder word vectors.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-disable_pretrained_emb', action="store_true",
                        help="Disable pretrained embedding (to make the training output log order consistent.")

    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    parser.add_argument('-log_interval', type=int, default=100,
                        help="logger.info stats at this interval.")

    parser.add_argument('-seed', type=int, default=-1,
                        help="""Random seed used for the experiments
                        reproducibility.""")
    parser.add_argument('-cuda_seed', type=int, default=-1,
                        help="""Random CUDA seed used for the experiments
                        reproducibility.""")

    parser.add_argument('-log_home', default='',
                        help="log home")

    parser.add_argument('-dump_epoch_checkpoint', action="store_true",
                        help="If want to store the checkpoint at each epoch")

    parser.add_argument('-continue_training', action="store_true", 
                        help="If checkpoint exist, then load latest checkpoint")

    # Data Management (loss mask)
    parser.add_argument('-use_good_data_only_before', default=-1, type=int,
                        help='Use good data to train before an epoch.')
    parser.add_argument('-add_all_bad_data_after', default=-1, type=int,
                        help='Use all data to train after an epoch. (only used in random_ratio mode)')
    
    parser.add_argument('-start_update_good_data_set', default=-1, type=int,
                        help='Start update good data set (< 0 means disable) (only used in threshold & magnitude mode)')
    parser.add_argument('-keep_initial_good_data_unchange', action="store_true", 
                        help="Don't update the initial good data set.")
    parser.add_argument('-early_stop_change_percent', type=float, default=0.01,
                        help='This will work in "threshold & magnitude" mode. When a certain percent of good data remains then stop. (< 0 means disable) (only used in threshold & magnitude mode)')

    parser.add_argument('-significant_criteria', type=str, default='random_ratio',
                        choices=['random_ratio', 'threshold', 'magnitude'],
                        help="""
                        random_ratio: add bad in ratio and will not update good data list
                        threshold & magnitude: if the 1st perdiction of sentences is significant then we set the data as good
                        threshold: if 1st prob greater than significant_value / paper_length
                        threshold: if 1st / 2st prob greater than significant_value
                        """)
    parser.add_argument('-significant_value', type=float, default=0.001,
                        help='Significant value used in "threshold" or "magnitude" significant good mode.')
