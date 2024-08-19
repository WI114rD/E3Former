## M
online_learning='full'
i=1
ns=(1 )
bszs=(1 )
lens=(60 30 10 1)
tasks=('FaaS_small' 'FaaS_c14' 'FaaS_medium' 'IaaS_c7' 'IaaS_c14' 'IaaS_c20' 'ETTh2_test')
methods=('mimo_patch_greedy' 'onenet_fsnet' 'mimo_patch' 'mimo_patchv1' 'fsnet' 'fsnet_time' 'itransformer' 'patch' 'timesnet' 'dlinear')
for n in ${ns[*]}; do
for bsz in ${bszs[*]}; do
for len in ${lens[*]}; do
for m in ${methods[*]}; do
for t in ${tasks[*]}; do
output_file="./results/output_${m}.out"
python3 -u main.py --method $m --root_path ./data/ --n_inner 1 --test_bsz 1 --data $t --features M --seq_len 1440 --label_len 0 --pred_len $len --des 'Exp' --itr 1 --train_epochs 10 --learning_rate 1e-3 --online_learning 'full' --use_adbfgs --freq 'h' --seed 2025 >> $output_file
done
done
done
done
done





