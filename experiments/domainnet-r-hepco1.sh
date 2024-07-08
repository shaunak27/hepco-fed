# bash experiments/imagenet-r.sh
# experiment settings\\


SPLIT=69
DATASET=IMBALANCEDOMAINNET
N_CLASS=345


# hard coded inputs
GPUID='0 1'
CONFIG_CLIP_P=configs/domainnet_clip_prompt.yaml
REPEAT=1
MEMORY=0
OVERWRITE=0
DEBUG=0

###############################################################

# process inputs
if [ $DEBUG -eq 1 ] 
then   
    MAXTASK=3
    SCHEDULE="2"
fi



DATE=hepco_v6.0_DOMAINNET_iid_cutoff_cutratio_0.4_seed_1_v1
OUTDIR=_outputs/${DATE}/${DATASET}/${SPLIT}-task
mkdir -p $OUTDIR
python -u run.py --config $CONFIG_CLIP_P --gpuid $GPUID --repeat $REPEAT --memory $MEMORY --overwrite $OVERWRITE --debug_mode $DEBUG \
    --learner_type prompt --learner_name L2P \
    --prompt_param 100 20 1 -1 --kl --hepco --imbalance 1 --percent 0.1 --n_clients 5 --n_rounds 10 --cutoff --lambda_KL 1 --cutoff_ratio 0.4 --replay_ratio 1 \
    --noise_dimension 64 --prompt_type weighted_l2p --wandb_name $DATE \
    --log_dir ${OUTDIR}/vit/l2p_multi-layer --overwrite 1 --seed 1 --lambda_mse 0.1
