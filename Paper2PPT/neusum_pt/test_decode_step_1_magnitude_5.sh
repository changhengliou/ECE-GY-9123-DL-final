#!/bin/bash
# For showing 3 sentences with score of each steps

export CUDA_VISIBLE_DEVICES=0 # same as the -gpus commend?!

DATAHOME="../../data" # this path is set in the https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling
QTYPE=${1:-"."}
MODELNAME=${2:-"IterativeLabeling_drop_data"}
SPECIFIC_EPOCH=${3:--1}

SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE

GLOVEPATH="../glove"

# currently you can only run in torch which compiled with cuda
python3 test.py -save_path $SAVEPATH \
                -specific_epoch $SPECIFIC_EPOCH \
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
                -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0 \
                -batch_size 1 -beam_size 10 -n_best_size 1 -output_len 5 \
                -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                -log_interval 100 -log_home $SAVEPATH \
                -seed 12345 -cuda_seed 12345 \
                -pre_word_vecs_enc $GLOVEPATH/glove.6B.50d.txt \
                -freeze_word_vecs_enc \
                -dev_input_src $DATAHOME/test/$QTYPE/test.txt.src \
                -dev_input_src_section $DATAHOME/test/$QTYPE/test.txt.section \
                -dev_ref $DATAHOME/test/$QTYPE/test.txt.oracle \
                -max_decode_step 5 -force_max_len \
                -stripping_mode magnitude -threshold 0.0001 \
                -set_postfix decode_step_1_magnitude_5
