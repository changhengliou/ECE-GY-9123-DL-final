QtypeToTrain=("future" "contribution" "baseline" "dataset")

SCRIPTNAME=${1:-"run.sh"}
MODELNAME=${2:-"RunAll$SCRIPTNAME"}

DATAHOME="../../data"
LOGPATH=$DATAHOME/models/$MODELNAME
mkdir -p $LOGPATH

# tensorboard --bind_all --logdir $LOGPATH --port 6006 &

for qtype in ${QtypeToTrain[@]}; do
    echo "Training $qtype"
    # run in sequential
    nohup bash $SCRIPTNAME $qtype $MODELNAME > /dev/null 2>&1

    # run all at once
    # nohup bash run.sh $qtype $MODELNAME > /dev/null 2>&1 &
done


# --host 0.0.0.0
# http://GCRAZGDL393:6006/
# tensorboard --bind_all --logdir $LOGPATH --port 6006
