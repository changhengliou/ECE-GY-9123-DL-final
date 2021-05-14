QtypeToTrain=("future" "contribution" "baseline" "dataset")

MODELNAME=${1:-"WEM_Start5_End10_IR10_AddBad_10-15"}

DATAHOME="../../data"
LOGPATH=$DATAHOME/models/$MODELNAME
mkdir -p $LOGPATH

# tensorboard --bind_all --logdir $LOGPATH --port 6006 &

for qtype in ${QtypeToTrain[@]}; do
    echo "Training $qtype"
    # run in sequential
    # nohup bash run.sh $qtype $MODELNAME > /dev/null 2>&1
    bash run.sh $qtype $MODELNAME | tee log.txt

    # run all at once
    # nohup bash run.sh $qtype $MODELNAME > /dev/null 2>&1 &
done


# --host 0.0.0.0
# http://GCRAZGDL393:6006/
# tensorboard --bind_all --logdir $LOGPATH --port 6006
