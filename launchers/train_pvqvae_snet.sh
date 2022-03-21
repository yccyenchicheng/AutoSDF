
RED='\033[0;31m'
NC='\033[0m' # No Color

gpuid=1
lr=1e-4
batch_size=6
max_dataset_size=10000000
dataset_mode='snet'
cat='all'
# cat='chair'
trunc_thres=0.2

nepochs=200
nepochs_decay=200

# model stuff
model='pvqvae'
vq_cfg='configs/pvqvae_snet.yaml'

# display stuff
display_freq=500 # default: 50 log every display_freq batches
print_freq=100 # default: 30

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

note="rerun"
name="${model}-${dataset_mode}-${cat}-LR${lr}-T${trunc_thres}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	batch_size=6
	max_dataset_size=12
    nepochs=100
    nepochs_decay=0
	update_html_freq=$(( 1 *"$batch_size" ))
	display_freq=$(( 1 *"$batch_size" ))
	print_freq=$(( 1 *"$batch_size" ))
    name="DEBUG-${name}"
fi


# CUDA_VISIBLE_DEVICES=${gpuid} python train.py --name ${name} --gpu_ids ${gpuid} --lr ${lr} --batch_size ${batch_size} \
python train.py --name ${name} --gpu_ids ${gpuid} --lr ${lr} --batch_size ${batch_size} \
                --model ${model} --vq_cfg ${vq_cfg} \
                --nepochs ${nepochs} --nepochs_decay ${nepochs_decay} \
                --dataset_mode ${dataset_mode} --cat ${cat} --trunc_thres ${trunc_thres} --max_dataset_size ${max_dataset_size} \
                --display_freq ${display_freq} --print_freq ${print_freq} \
                 --debug ${debug}
