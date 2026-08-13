
corruption_list=( "missing" "video" "audio")

for bs in 16 ; do
for wc in  0.01; do
for wg in   1  ; do
for gam in 0.1 1; do
for alpha in  0.95 ; do
for beta in  0.5; do
for temp in 1; do
for warmup_a in 1000; do
for warmup_v in 100 ; do
for corruption in "${corruption_list[@]}"; do
for seed in 111   ; do

    python run_adapgc.py \
        --dataset 'ks50' \
        --json-root 'json_csv_files/ks50/' \
        --label-csv 'json_csv_files/class_labels_indices_ks50.csv' \
        --pretrain_path 'checkpoints/cav_mae_ks50.pth' \
        --tta-method 'ADAPGC' \
        --severity-start 5 \
        --severity-end 5 \
        --corruption-modality ${corruption} \
        --batch-size ${bs} \
        --gpu 0 \
        --w-c ${wc} \
        --w-g ${wg} \
        --gamma ${gam} \
        --alpha ${alpha} \
        --beta ${beta} \
        --temp ${temp} \
        --warmup-a ${warmup_a} \
        --warmup-v ${warmup_v} \
        --seed ${seed} \
        --exp-name Ours-seed${seed}-alpha${alpha}-Predlogi-gam${gam}-wg${wg}-GDA${beta}-at${temp}-supAll-ddual-warmupa${warmup_a}v${warmup_v}

done
done
done 
done
done
done
done
done
done
done
done













# # corruption_list=("missing"  "audio"  "video" "single"  "clean")


# corruption_list=( "missing" )


# for bs in 16 ; do
# for wc in  0.01; do
# for wg in   1  ; do
# for gam in  1  ; do
# for alpha in  0.95 ; do
# for beta in  0.5 1; do
# for temp in 1; do
# for warmup_a in 1000; do
# for warmup_v in 100 ; do

# for corruption in "${corruption_list[@]}"; do
#     python run_adapgc.py \
#         --dataset 'ks50' \
#         --json-root 'json_csv_files/ks50/' \
#         --label-csv 'json_csv_files/class_labels_indices_ks50.csv' \
#         --pretrain_path 'checkpoints/cav_mae_ks50.pth' \
#         --tta-method 'ADAPGC' \
#         --severity-start 5 \
#         --severity-end 5 \
#         --corruption-modality ${corruption} \
#         --batch-size ${bs} \
#         --gpu 6,7 \
#         --w-c ${wc} \
#         --w-g ${wg} \
#         --gamma ${gam} \
#         --alpha ${alpha} \
#         --beta ${beta} \
#         --temp ${temp} \
#         --warmup-a ${warmup_a} \
#         --warmup-v ${warmup_v} \
#         --exp-name Ours-alpha${alpha}-Predlogi-GDA${beta}-at${temp}-supAll-ddual-warmupa${warmup_a}v${warmup_v}

# done
# done
# done 
# done
# done
# done
# done
# done
# done
# done



