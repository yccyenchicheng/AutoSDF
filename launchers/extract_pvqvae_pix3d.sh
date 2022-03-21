gpuid=1

# datastuff
batch_size=1
dataset_mode='pix3d'
trunc_thres=0.2

# model stuff
model='pvqvae'

cat='all'

# ours
vq_cfg='configs/pvqvae_snet.yaml'
ckpt='saved_ckpt/pvqvae-snet-all-LR1e-4-T0.2-rerun-epoch140.pth'
vq_note='default'

today=$(date '+%m%d')
me=`basename "$0"`
me=$(echo $me | cut -d'.' -f 1)

# CUDA_VISIBLE_DEVICES=${gpuid} python train.py --name ${name} --gpu_ids ${gpuid} --lr ${lr} --batch_size ${batch_size} \
python extract_code.py --gpu_ids ${gpuid} --batch_size ${batch_size} \
                    --model ${model} --vq_cfg ${vq_cfg} --vq_note ${vq_note} --ckpt ${ckpt} \
                    --dataset_mode ${dataset_mode} --cat ${cat} --trunc_thres ${trunc_thres} --serial_batches \
