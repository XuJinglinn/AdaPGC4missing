
corruption_list=( "missing")
for corruption in "${corruption_list[@]}"; do
    python run_adapgc.py \
        --dataset 'ks50' \
        --json-root 'json_csv_files/ks50/' \
        --label-csv 'json_csv_files/class_labels_indices_ks50.csv' \
        --pretrain_path 'checkpoints/cav_mae_ks50.pth' \
        --tta-method 'ADAPGC' \
        --severity-start 5 \
        --severity-end 5 \
        --corruption-modality ${corruption} \
        --batch-size 16 \
        --gpu '6,7' \
        --w-c 0.01 \
        --w-g 1 \
        --gamma 1 \
        --exp-name "run_missing_dual_adapgc"
done



for corruption in "${corruption_list[@]}"; do
    python run_adapgc.py \
        --dataset 'vggsound' \
        --json-root 'json_csv_files/vgg/' \
        --label-csv 'json_csv_files/class_labels_indices_vgg.csv' \
        --pretrain_path 'checkpoints/vgg_65.5.pth' \
        --tta-method 'ADAPGC' \
        --severity-start 5 \
        --severity-end 5 \
        --corruption-modality ${corruption} \
        --batch-size 16 \
        --gpu '6,7' \
        --w-c 0.01 \
        --w-g 1 \
        --gamma 1 \
        --exp-name "run_missing_dual_adapgc"
done

