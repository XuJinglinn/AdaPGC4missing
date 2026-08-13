

corruption_list=( "single" "missing" "clean" "audio" "video")



for bs in 32 64 128; do
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
        --batch-size ${bs} \
        --gpu 6,7 \
        --w-c 0.01 \
        --w-g 1 \
        --gamma 1 \
        --w-read 1 \
        --exp-name adapgc_${corruption}_bs${bs}_dual
done
done


for bs in 32 64 128; do
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
        --batch-size ${bs} \
        --gpu 4,5 \
        --w-c 0.01 \
        --w-g 1 \
        --gamma 1 \
        --w-read 1 \
        --exp-name adapgc_${corruption}_bs${bs}_dual
done
done



