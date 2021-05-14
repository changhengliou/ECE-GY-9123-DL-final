#!/bin/bash

# If use a name without log-linear checkpoints, it will evaluate on default weight (in eval_loglinear.sh)
MODELNAME=${1:-"LOGLINEAR_TEST_ONLY"}

DATAHOME="../../data"
SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE

LOGPATH=$DATAHOME/models/$MODELNAME

# QtypeToTest=("future" "contribution" "baseline" "dataset" "motivation" "metric")
# QtypeToTest=("future" "contribution" "baseline" "dataset" "motivation")
QtypeToTest=("contribution" "dataset" "baseline" "future")

TestToRun=("eval_loglinear.sh")

models_string=$(printf "_%s" "${QtypeToTest[@]}")
test_string=$(printf "_%s" "${TestToRun[@]}")
LOGFILE=$LOGPATH/"ResultOfQtypes$models_string$test_string.txt"
mkdir -p $LOGPATH
touch $LOGFILE

for test_script in ${TestToRun[@]}; do
    echo "================== Test all topics with $test_script =================="
    for qtype in ${QtypeToTest[@]}; do
        echo "Testing $qtype"
        performance=`bash $test_script $qtype $MODELNAME |& tail -n 1`
        echo "$performance"
        echo "$QTYPE $test_script: $performance" >> $LOGFILE
    done
done

# echo "======================================================================="
# echo "Show all at once:"
# cat $LOGFILE
