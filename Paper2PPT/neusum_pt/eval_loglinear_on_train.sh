#!/bin/bash

# TODO: currently deprecated

export CUDA_VISIBLE_DEVICES=0 # same as the -gpus commend?!

DATAHOME="../../data" # this path is set in the https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling
QTYPE=${1:-"."}
MODELNAME=${2:-"."}
SPECIFIC_EPOCH=${3:--1}

SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE
# SAVEPATH=$DATAHOME/loglinear/$MODELNAME/$QTYPE
mkdir -p $SAVEPATH

GLOVEPATH="../glove"

# currently you can only run in torch which compiled with cuda
# delete "-gpus 0 \" to run on cpu
# delete "-disable_pretrained_emb \" to use pre-trained embedding; otherwise delete "-freeze_word_vecs_enc \"
# set n of "-relabel_epoch n" higher than epochs to disable iterative labeling
# set "-curriculum 0 -extra_shuffle \" to shuffle training data every epoch
#  -section_embedding
python3 eval_loglinear.py -save_path $SAVEPATH \
                          -specific_epoch $SPECIFIC_EPOCH \
                          -set_postfix _on_train \
                          -online_process_data \
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
                          -batch_size 4 -beam_size 10 -n_best_size 1 -output_len 10 \
                          -epochs 20 \
                          -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                          -start_eval_batch 1000 -eval_per_batch 1000 \
                          -log_interval 100 -log_home $SAVEPATH \
                          -seed 12345 -cuda_seed 12345 \
                          -pre_word_vecs_enc $GLOVEPATH/glove.6B.50d.txt \
                          -freeze_word_vecs_enc \
                          -dev_input_src $DATAHOME/train/$QTYPE/train.txt.src \
                          -dev_input_src_section $DATAHOME/train/$QTYPE/train.txt.section \
                          -dev_ref $DATAHOME/train/$QTYPE/train.txt.oracle \
                          -position_weight 0 -keyword_weight 1 -in_bert_weight 1 -in_section_weight 1 -qtype $QTYPE \
                          -max_decode_step 1 -force_max_len
