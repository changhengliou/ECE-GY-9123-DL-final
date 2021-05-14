#!/bin/bash

export CUDA_VISIBLE_DEVICES=0 # same as the -gpus commend?!

DATAHOME="../../data" # this path is set in the https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling
QTYPE=${1:-"."}
MODELNAME=${2:-"Ours"}
# LOG_LINEAR_MODELNAME=${3:-"."}

SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE
LOG_LINEAR_SAVEPATH=$SAVEPATH
# LOG_LINEAR_SAVEPATH=$DATAHOME/loglinear/$LOG_LINEAR_MODELNAME/$QTYPE
# LOG_LINEAR_SAVEPATH=$DATAHOME/loglinear/$QTYPE
mkdir -p $SAVEPATH

GLOVEPATH="../glove"

# currently you can only run in torch which compiled with cuda
# delete "-gpus 0 \" to run on cpu
# delete "-disable_pretrained_emb \" to use pre-trained embedding; otherwise delete "-freeze_word_vecs_enc \"
# set n of "-relabel_epoch n" higher than epochs to disable iterative labeling
# set "-curriculum 0 -extra_shuffle \" to shuffle training data every epoch
# To load pre-trained log-linear model set these two arguments -log_linear_save_path $LOG_LINEAR_SAVEPATH -qtype $QTYPE
# Currently, when using log-linear model, make sure the max_decode_step is set to 1, or there might cause error when iterative labeling

# If eval score on dev use this -start_eval_batch 1000 -eval_per_batch 1000 \
# If eval loss on dev use this -start_eval_batch 0 -eval_per_batch 100 \
# If dev set size is 0, then delete this

# Section title feature
# 1. to use average embedding just add "-section_embedding", but the log-linear model better be pre-trained
# 2. to use binary embedding just add "-in_section_weight {int > 0}", better set batch size to <= 32 or it might cause memory issue (reduce -dec_rnn_size won't help here)

python3 train.py -save_path $SAVEPATH \
                 -online_process_data \
                 -continue_training \
                 -max_doc_len 500 \
                 -drop_too_short 50 -drop_too_long 500 \
                 -train_oracle $DATAHOME/train/$QTYPE/train.txt.oracle \
                 -train_src $DATAHOME/train/$QTYPE/train.txt.src \
                 -train_src_section $DATAHOME/train/$QTYPE/train.txt.section \
                 -layers 1 -word_vec_size 50 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
                 -sent_brnn -doc_brnn \
                 -dec_init simple \
                 -att_vec_size 256 \
                 -norm_lambda 20 \
                 -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
                 -batch_size 64 -beam_size 10 -n_best_size 1 -output_len 5 \
                 -gpus 0 \
                 -epochs 20 \
                 -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                 -start_eval_batch 0 -eval_per_batch 100 \
                 -log_interval 100 -log_home $SAVEPATH \
                 -seed 12345 -cuda_seed 12345 \
                 -pre_word_vecs_enc $GLOVEPATH/glove.6B.50d.txt \
                 -dev_input_src $DATAHOME/dev/$QTYPE/dev.txt.src \
                 -dev_input_src_section $DATAHOME/dev/$QTYPE/dev.txt.section \
                 -dev_ref $DATAHOME/dev/$QTYPE/dev.txt.oracle \
                 -test_input_src $DATAHOME/test/$QTYPE/test.txt.src \
                 -test_input_src_section $DATAHOME/test/$QTYPE/test.txt.section \
                 -test_ref $DATAHOME/test/$QTYPE/test.txt.oracle \
                 -test_bert_annotation $DATAHOME/test/$QTYPE/test.txt.bert \
                 -eval_test_during_train \
                 -max_decode_step 1 -force_max_len \
                 -stripping_mode none -threshold 0.0001 \
                 -dump_epoch_checkpoint \
                 -relabel_epoch 9 -start_train_with_em 5 \
                 -stop_em_and_loglinear_after 10 \
                 -use_good_data_only_before 10 -add_all_bad_data_after 15 \
                 -significant_criteria random_ratio \
                 -enable_log_linear -loglinear_learning_rate 0.01 -position_weight 0 -keyword_weight 1 -in_bert_weight 1 -in_section_weight 1 -qtype $QTYPE
