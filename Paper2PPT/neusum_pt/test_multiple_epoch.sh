#!/bin/bash

QTYPE=${1:-"."}
MODELNAME=${2:-"IterativeLabeling_drop_data"}
# EPOCHS=(1 2 3 5 10 15 20 25 30)
EPOCHS=($(seq 1 1 20))

DATAHOME="../../data" # this path is set in the https://dev.azure.com/v-dawle/_git/BERTIterativeLabeling
SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE

# https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
epochs_string=$(printf "_%s" "${EPOCHS[@]}")
LOGFILE=$SAVEPATH/"ResultOfMultipleEpoch$epochs_string.txt"
touch $LOGFILE

# If don't use log-linear model, just set this to false
USE_LOGLINEAR=true

# https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays
for epoch in ${EPOCHS[@]}; do
    # https://askubuntu.com/questions/621681/how-can-i-execute-the-last-line-of-output-in-bash
    echo "Testing epoch $epoch"
    performance=`bash test.sh $QTYPE $MODELNAME $epoch |& tail -n 1`
    echo "Epoch $epoch: $performance" >> $LOGFILE

    if [ "$USE_LOGLINEAR" ]; then
        # https://stackoverflow.com/questions/13832866/unix-show-the-second-line-of-the-file
        loglinear_performance_temp=`bash eval_loglinear.sh $QTYPE $MODELNAME $epoch |& tail -n 4`
        loglinear_performance=`echo $loglinear_performance_temp | tail -n 1`
        loglinear_weight=`echo $loglinear_performance_temp | head -n 1`
        echo "Epoch $epoch (log-linear): $loglinear_performance" >> $LOGFILE
        echo "Epoch $epoch: $loglinear_weight" >> $LOGFILE
    fi
done

cat $LOGFILE
