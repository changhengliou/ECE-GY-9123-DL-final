# Run different settings on a single question type

BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'
MODELHOME="../../data/models"

QTYPE=${1:-"future"}
MODELPREFIX="${2:-RunAll}"
TENSORBOARD_TEMP="$MODELHOME/$MODELPREFIX-TENSORBOARD_TEMP"

# SETTINGTORUN=("run_ours.sh" "run_without_IL.sh" "run_IL_only.sh" "run_update_good_magnitude.sh" "run_update_good_threshold.sh")
# SETTINGTORUN=("run_ours.sh" "run_without_IL.sh" "run_update_good_magnitude.sh" "run_update_good_threshold.sh") # models involves the log-linear model
# SETTINGTORUN=("run_ours_loglinear_0001.sh" "run_ours_loglinear_0010.sh" "run_ours_loglinear_0100.sh") # different initial weight of log-linear model
SETTINGTORUN=("run_ours.sh" "run_without_IL.sh")

mkdir -p $TENSORBOARD_TEMP
# tensorboard --bind_all --logdir $TENSORBOARD_TEMP --port 6006 &

for script in ${SETTINGTORUN[@]}; do
    MODELNAME="$MODELPREFIX-$script"
    MODELPATH="$MODELHOME/$MODELNAME/$QTYPE"
    TBTEMPPATH="$TENSORBOARD_TEMP/$QTYPE-$script"
    echo -e "${RED}Running $script $QTYPE $MODELNAME.....${NC}"

    if ! test -e $MODELPATH; then
        echo -e "${BLUE}Create soft link from $MODELREALPATH to $TBTEMPPATH ${NC}"
        mkdir -p $MODELPATH
        MODELREALPATH=`realpath $MODELPATH`
        ln -s $MODELREALPATH $TBTEMPPATH
    else
        echo -e "${BLUE}Soft link already exist at $TBTEMPPATH ${NC}"
    fi

    bash $script $QTYPE $MODELPREFIX-$script
    echo -e "${BLUE}You can checkout result at $MODELPATH${NC}"
done
