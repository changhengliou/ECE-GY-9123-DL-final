# Run different settings on a single question type

BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'
DATAHOME="../../data/models"

QTYPE=${1:-"future"}
MODELPREFIX="${2:-RunAll}"
TENSORBOARD_TEMP="$DATAHOME/$MODELPREFIX-TENSORBOARD_TEMP"

# SETTINGTORUN=("run_ours.sh" "run_without_IL.sh" "run_IL_only.sh" "run_update_good_magnitude.sh" "run_update_good_threshold.sh")
SETTINGTORUN=("run_ours.sh" "run_without_IL.sh" "run_update_good_magnitude.sh" "run_update_good_threshold.sh") # models involves the log-linear model

mkdir -p $TENSORBOARD_TEMP

for script in ${SETTINGTORUN[@]}; do
    MODELNAME="$MODELPREFIX-$script"
    MODELPATH="$DATAHOME/$MODELNAME/$QTYPE"
    TBTEMPPATH="$TENSORBOARD_TEMP/$QTYPE-$script"
    echo -e "${RED}Running $script $QTYPE $MODELNAME.....${NC}"
    mkdir -p $MODELPATH
    MODELREALPATH=`realpath $MODELPATH`
    # TBTEMPREALPATH=`realpath $TENSORBOARD_TEMP`
    # ln -s $MODELREALPATH $TENSORBOARD_TEMP
    # mv $TENSORBOARD_TEMP/$QTYPE $TBTEMPPATH
    # echo -e "${BLUE}Create soft link from $MODELREALPATH to $TBTEMPREALPATH ${NC}"
    ln -s $MODELREALPATH $TBTEMPPATH
    echo -e "${BLUE}Create soft link to $TBTEMPPATH ${NC}"
    #bash $script $QTYPE $MODELPREFIX-$script
    echo -e "${BLUE}You can checkout result at $MODELPATH${NC}"
done


tensorboard --bind_all --logdir $TENSORBOARD_TEMP --port 6099
