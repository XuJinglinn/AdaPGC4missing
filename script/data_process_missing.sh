
# 部分missing

# python ./data_process/create_missing_json.py \
#         --clean-path 'json_csv_files/ks50/clean/severity_0.json' \
#         --missing-video-path '/data/xjl/dataset/Kinetics50/image_mulframe_val256_k=50-C/missing/severity_5' \
#         --missing-audio-path '/data/xjl/dataset/Kinetics50/audio_val256_k=50-C/missing/severity_5' \
#         --missing-type 'all' \
#         --missing-rate 'all' \
#         --output-dir 'json_csv_files/ks50/missing'

# python ./data_process/create_missing_json.py \
#     --clean-path 'json_csv_files/vgg/clean/severity_0.json' \
#     --missing-video-path '/data/xjl/dataset/VGGSound/image_mulframe_test-C/missing/severity_5' \
#     --missing-audio-path '/data/xjl/dataset/VGGSound/audio_test-C/missing/severity_5' \
#     --missing-type 'all' \
#     --missing-rate 'all' \
#     --output-dir 'json_csv_files/vgg/missing'
    
# 全missing

python ./data_process/create_missing_json.py \
        --clean-path 'json_csv_files/ks50/clean/severity_0.json' \
        --missing-video-path '/data/xjl/dataset/Kinetics50/image_mulframe_val256_k=50-C/missing/severity_5' \
        --missing-audio-path '/data/xjl/dataset/Kinetics50/audio_val256_k=50-C/missing/severity_5' \
        --missing-type a \
        --missing-rate 1.0 \
        --output-dir 'json_csv_files/ks50/single'

python ./data_process/create_missing_json.py \
        --clean-path 'json_csv_files/ks50/clean/severity_0.json' \
        --missing-video-path '/data/xjl/dataset/Kinetics50/image_mulframe_val256_k=50-C/missing/severity_5' \
        --missing-audio-path '/data/xjl/dataset/Kinetics50/audio_val256_k=50-C/missing/severity_5' \
        --missing-type v \
        --missing-rate 1.0 \
        --output-dir 'json_csv_files/ks50/single'

python ./data_process/create_missing_json.py \
    --clean-path 'json_csv_files/vgg/clean/severity_0.json' \
    --missing-video-path '/data/xjl/dataset/VGGSound/image_mulframe_test-C/missing/severity_5' \
    --missing-audio-path '/data/xjl/dataset/VGGSound/audio_test-C/missing/severity_5' \
    --missing-type a \
    --missing-rate 1.0 \
    --output-dir 'json_csv_files/vgg/single'

python ./data_process/create_missing_json.py \
    --clean-path 'json_csv_files/vgg/clean/severity_0.json' \
    --missing-video-path '/data/xjl/dataset/VGGSound/image_mulframe_test-C/missing/severity_5' \
    --missing-audio-path '/data/xjl/dataset/VGGSound/audio_test-C/missing/severity_5' \
    --missing-type v \
    --missing-rate 1.0 \
    --output-dir 'json_csv_files/vgg/single'