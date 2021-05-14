#!/bin/bash

MODELNAME=${1:-"WEM_Start5_End10_IR10_AddBad_10-15"}

DATAHOME="../../data"
SAVEPATH=$DATAHOME/models/$MODELNAME/$QTYPE

LOGPATH=$DATAHOME/models/$MODELNAME

# https://stackoverflow.com/questions/22727107/how-to-find-the-last-field-using-cut/22727211
# QtypeToTest=("future" "contribution" "baseline" "dataset" "motivation" "metric")
# QtypeToTest=($(cd $LOGPATH; ls -l | grep '^d' | grep -o '[^ ]*$'))
QtypeToTest=("contribution" "dataset" "baseline" "future")

# TestToRun=("test.sh")
# TestToRun=("test_decode_step_1.sh" "test_decode_step_3_beam_search.sh" "test_decode_step_1_topk_3.sh" "test_decode_step_1_threshold_3.sh" "test_decode_step_1_magnitude_3.sh")
# TestToRun=($(ls test_decode_step*.sh))
# TestToRun=("test_decode_step_1.sh" "test_decode_step_1_topk_3.sh") # Experiment Table
TestToRun=("test_decode_step_1.sh" "test_decode_step_1_topk_3.sh" "test_decode_step_1_diff_3.sh")

models_string=$(printf "_%s" "${QtypeToTest[@]}")
test_string=$(printf "_%s" "${TestToRun[@]}")
LOGFILE=$LOGPATH/"ResultOfQtypes$models_string$test_string.txt"
touch $LOGFILE

if (( ${#QtypeToTest[@]} == 0 )); then
    echo "Can't find qtype under $LOGPATH"
    exit
fi

for test_script in ${TestToRun[@]}; do
    echo "================== Test all topics with $test_script =================="
    for qtype in ${QtypeToTest[@]}; do
        echo "Testing $qtype"
	bash $test_script $qtype $MODELNAME |& tee log_$qtype.txt
        # performance="bash $test_script $qtype $MODELNAME |& tail -n 1"
        # echo "$performance"
        #echo "$QTYPE $test_script: $performance" >> $LOGFILE
    done
done

# echo "======================================================================="
# echo "Show all at once:"
# cat $LOGFILE
