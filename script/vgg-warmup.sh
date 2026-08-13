



# corruption_list=("missing"  "audio"  "video" "single"  "clean")


corruption_list=( "missing" )

for bs in 16 ; do
for wc in  0.01; do
for wg in   1 ; do
for gam in  1; do
for alpha in  0.95 ; do
for beta in  0.5 1; do
for temp in 1; do
for warmup_a in  1500 2000; do
for warmup_v in 1000 1500 2000; do
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
        --w-c ${wc} \
        --w-g ${wg} \
        --gamma ${gam} \
        --alpha ${alpha} \
        --temp ${temp} \
        --warmup-a ${warmup_a} \
        --warmup-v ${warmup_v} \
        --exp-name testMem-Ours-alpha${alpha}-Predlogi-gam${gam}-GDA${beta}-at${temp}-supAll-ddual-warmupa${warmup_a}v${warmup_v}


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

