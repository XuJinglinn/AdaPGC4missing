
# corruption_list=("missing"  "audio"  "video" "single"  "clean")


corruption_list=(  "audio"  "video"  )


for bs in 16 ; do
for wc in  0.01 0.1 1; do
for wg in   0.1 1 10  ; do
for gam in  0.1 1 10  ; do
for alpha in  0.95 ; do
for beta in  0 ; do
for temp in 1; do
# for n0 in 100 1000 1500 2000 5000; do
for corruption in "${corruption_list[@]}"; do
    python run_diag_adapgc.py \
        --dataset 'ks50' \
        --json-root 'json_csv_files/ks50/' \
        --label-csv 'json_csv_files/class_labels_indices_ks50.csv' \
        --pretrain_path 'checkpoints/cav_mae_ks50.pth' \
        --tta-method 'ADAPGC' \
        --severity-start 5 \
        --severity-end 5 \
        --corruption-modality ${corruption} \
        --batch-size ${bs} \
        --gpu 0,1 \
        --w-c ${wc} \
        --w-g ${wg} \
        --gamma ${gam} \
        --alpha ${alpha} \
        --beta ${beta} \
        --temp ${temp} \
        --exp-name Ours-diag-wc${wc}-wg${wg}-gamma${gam}

done
done
done 
done

done
done
# done
done
done


