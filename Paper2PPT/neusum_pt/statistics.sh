#!/bin/bash
# Just some dummy parameters

DATAHOME="../../data" # this path is set in the https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling
QTYPE=${1:-"."}
MODELNAME=${2:-"Ours"}

SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE
LOG_LINEAR_SAVEPATH=$SAVEPATH
mkdir -p $SAVEPATH

GLOVEPATH="../glove"

python3 statistics.py -save_path $SAVEPATH \
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
                      -epochs 15 \
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
                      -relabel_epoch 99 -start_train_with_em 3 \
                      -stop_em_and_loglinear_after 99 \
                      -use_good_data_only_before 15 -add_all_bad_data_after 99 \
                      -significant_criteria threshold -significant_value 0.5 \
                      -start_update_good_data_set 6 -keep_initial_good_data_unchange -early_stop_change_percent 0.01 \
                      -enable_log_linear -position_weight 0 -keyword_weight 1 -in_bert_weight 1 -in_section_weight 1 -qtype $QTYPE