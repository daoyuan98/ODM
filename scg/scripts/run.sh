buffer_size=64
Ks=4
decays=(0.2)

cache_dir=/path/to/cache
log_file=/path/to/log

if [ ! -d ${cache_dir} ] 
then
    mkdir -p ${cache_dir}
fi

if [ ! -f ${log_file} ]
then
    touch ${log_file}
fi

python -u main_scg.py \
    --world-size 2 \
    --cache-dir ${cache_dir} \
    --batch-size 2 \
    --num-epochs 12 \
    --balance-classifier-weight 1.0 \
    --consistency-weight 0.00 \
    --lr-decay ${decay} \
    --print-interval 200 \
    --learning-rate 0.0001 \
    --buffer-size ${buffer_size} \
    --K ${K} \
    --max-human 15 \
    --max-object 15 \
    --age-weight 0.5 \
    --start-balance-epoch 3 \
    --start-consistency-epoch 3  

