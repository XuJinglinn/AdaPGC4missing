

corruption_list=( "single" "missing" "clean" "audio" "video")
# corruption_list=( "clean" )




# for bs in 16 ; do
# for wc in 0 0.01 ; do
# for wg in 0 0.1 1 10; do
# for gam in 0 0.1 1 10; do
# for corruption in "${corruption_list[@]}"; do
#     python run_adapgc.py \
#         --dataset 'vggsound' \
#         --json-root 'json_csv_files/vgg/' \
#         --label-csv 'json_csv_files/class_labels_indices_vgg.csv' \
#         --pretrain_path 'checkpoints/vgg_65.5.pth' \
#         --tta-method 'ADAPGC' \
#         --severity-start 5 \
#         --severity-end 5 \
#         --corruption-modality ${corruption} \
#         --batch-size ${bs} \
#         --gpu 6,7 \
#         --w-c ${wc} \
#         --w-g ${wg} \
#         --gamma ${gam} \
#         --exp-name adapgc_${corruption}_bs${bs}_wc${wc}_wg${wg}_gam${gam}_dual_rmREAD
# done
# done
# done 
# done
# done




for bs in 16 ; do
for wc in 0 0.01 ; do
for wg in 0 0.1 1 10; do
for gam in 0 0.1 1 10; do
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
        --gpu 6,7 \
        --w-c ${wc} \
        --w-g ${wg} \
        --gamma ${gam} \
        --exp-name adapgc_${corruption}_bs${bs}_wc${wc}_wg${wg}_gam${gam}_dual_rmREAD
done
done
done 
done
done





