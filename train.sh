#python3.5 main.py --branch_combining_mode 'concat' --task_name 'multi_makeup_branch_size=7_branch_combining_mode=concat' --batch_size 6 --batch_size_stage2 10

#python3.5 main.py --branch_combining_mode 'add' --task_name 'multi_makeup_branch_size=7_branch_combining_mode=concat' --batch_size 6 --batch_size_stage2 10
# python3.5 main.py --branch_combining_mode 'concat' --task_name 'imlevel_concatGAN_makeup' --batch_size 1 \
# --MB_flag True --only_concat True --mode 'evaluate'
#python3.5 main.py --branch_combining_mode 'concat' --task_name 'MBGAN_makeup' --batch_size 11 --mode 'train'
#python3.5 main.py  --task_name 'attention_mbgan_makeup' --batch_size 11 --mode 'train'

# python3.5 main.py  --task_name 'MBGAN_yamaha_300' --batch_size 11 --mode 'train' --dataset2_image_path './data/yamaha-aligned-5cls' --c_dim 5 --c2_dim 5 --skip_act 'sigmoid' --skip_mode 'add' \
# --num_epochs 400 --num_epochs_decay 200 --pretrained_model 120_400

# python3.5 main.py  --task_name 'MBGAN_CFEE' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' --c_dim 22 --c2_dim 22 --skip_act 'sigmoid' --skip_mode 'add' \
# --num_epochs 200 --num_epochs_decay 100

# python3.5 main.py  --task_name 'MGAN_yamaha' --batch_size 1 --mode 'vis' --dataset2_image_path './data/yamaha-aligned-5cls'
# python3.5 main.py  --task_name 'MGAN_makeup' --batch_size 1 --mode 'vis' --dataset2_image_path './data/makeup-data5.0-big-YZ-v2'

# python3.5 main.py  --task_name 'standar_MBGAN_CFEE' --batch_size 11 --mode 'train' --dataset2_image_path './data/yamaha-aligned-5cls' --c_dim 5 --c2_dim 5 --skip_act 'relu' --skip_mode 'concat' \
# --num_epochs 200 --num_epochs_decay 100 --pretrained_model 146_400

# python3.5 main.py  --task_name 'MBGAN_CFEE' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' --c_dim 22 --c2_dim 22 --skip_act 'sigmoid' --skip_mode 'add' \
# --num_epochs 200 --num_epochs_decay 100

# python3.5 main.py  --task_name 'reluMGAN_CFEE' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' --c_dim 22 --c2_dim 22 --skip_act 'relu' --skip_mode 'add' \
# --num_epochs 200 --num_epochs_decay 100

# python3.5 main.py  --task_name 'MBGAN_celebA' --batch_size 11 --mode 'train' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --dataset1_crop_size 256 --resume '16_16600'

# python3.5 main.py  --task_name 'MBGAN_makeup_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path './data/makeup-data5.0-big-YZ-v2' --pretrained_model 'MBGAN_celebA/models/16_16600'
# python3.5 main.py  --task_name 'MBGAN_makeup_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path './data/makeup-data5.0-big-YZ-v2'
# python3.5 main.py  --task_name 'MGAN_celebA' --batch_size 1 --mode 'evaluate' --test_iters 15200 --dataset 'CelebA'
# python3.5 main.py  --task_name 'MGAN_yamaha_pretrain_withoutD' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/yamaha-aligned-5cls' --pretrained_model 'MGAN_celebA/models/20_15200'
# python3.5 main.py  --task_name 'MGAN_yamaha_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/yamaha-aligned-5cls'

#  python3.5 main.py  --task_name 'MBGAN_yamaha_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/yamaha-aligned-5cls' --pretrained_model 'MBGAN_celebA/models/16_16600'
# python3.5 main.py  --task_name 'MBGAN_cfee_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' --pretrained_model 'MBGAN_celebA/models/20_16600' \
# --c_dim 22 --c2_dim 22
# python3.5 main.py  --task_name 'MBGAN_yamaha_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/yamaha-aligned-5cls'
# python3.5 main.py  --task_name 'MBGAN_oulu_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/oulu' --pretrained_model 'MBGAN_celebA/models/20_16600' \
# --c_dim 6 --c2_dim 6

# python3.5 main.py  --task_name 'MBGAN_cfee_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' \
# --c_dim 22 --c2_dim 22

# python3.5 main.py  --task_name 'MGAN_cfee_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' --pretrained_model 'MGAN_celebA/models/19_15200' \
# --c_dim 22 --c2_dim 22

# python3.5 main.py  --task_name 'MBGAN_oulu_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/oulu' \
# --c_dim 6 --c2_dim 6
# python3.5 main.py  --task_name 'MBGAN_makeup_om' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' \
# --c_dim 3 --c2_dim 3 --model_save_step 347 --sample_step 347

# python3.5 main.py  --task_name 'MBGAN_makeup_om_pretrain' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' --pretrained_model 'MBGAN_celebA/models/20_16600' \
# --c_dim 3 --c2_dim 3 --model_save_step 347 --sample_step 347

# python3.5 main.py  --task_name 'pretrain_MBGAN_illumination' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/illumination_big_v1' --pretrained_model 'MBGAN_celebA/models/20_16600' \
# --c_dim 19 --c2_dim 19 --model_save_step 1459 --sample_step 1459

# python3.5 main.py  --task_name 'MGAN_makeup_om' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' \
# --c_dim 3 --c2_dim 3 --model_save_step 347 --sample_step 347
# python3.5 main.py  --task_name 'MBGAN_makeup_om_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' \
# --c_dim 3 --c2_dim 3 --model_save_step 347 --sample_step 347

# python3.5 main.py  --task_name 'MGAN_cfee_pretrain' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CFEE256' \
# --c_dim 22 --c2_dim 22

# python3.5 main.py  --task_name 'MGAN_oulu' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/oulu' \
# --c_dim 6 --c2_dim 6

# python3.5 main.py  --task_name 'MGAN_makeup_om' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' \
# --c_dim 3 --c2_dim 3
# python3.5 main.py  --task_name 'MBGAN_makeup_om' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/makeup-data5.0-big-OM-v3' \
# --c_dim 3 --c2_dim 3

# python3.5 main.py  --task_name 'MGAN_oulu_pretrain' --batch_size 11 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/oulu'  --pretrained_model 'MGAN_celebA/models/20_15200'\
# --c_dim 6 --c2_dim 6

# python3.5 main.py  --task_name 'MGAN_celebA' --batch_size 1 --mode 'evaluate' --test_iters 15200 --dataset 'CelebA'

# python3.5 main.py  --task_name 'MBGAN_celebA_100' --batch_size 11 --mode 'train' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 100 --num_epochs_decay 50 --log_step 1000 --sample_step 16600 --model_save_step 16600 --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MBGAN_illumination' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/illumination_big_v1' --pretrained_model '' \
# --c_dim 19 --c2_dim 19 --model_save_step 1459 --sample_step 1459

# python3.5 main.py  --task_name 'MBGAN_illumination' --batch_size 1 --mode 'evaluate' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/illumination_big_v1' --pretrained_model '' \
# --c_dim 19 --c2_dim 19 --model_save_step 1459 --sample_step 1459

# python3.5 main.py  --task_name 'MBGAN_celebA' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MGAN_celebA' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MBGAN_celebA' --batch_size 11 --mode 'train' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MBGAN_celebA_1_128' --batch_size 22 --mode 'train' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 100 --sample_step 8000 --model_save_step 8300 --dataset1_crop_size 256 --image_size 128

# python3.5 main.py  --task_name 'MBGAN_celebA' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --dataset1_crop_size 256 --test_label_path /home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/test_label_list.txt

# python3.5 main.py  --task_name 'MBGAN_celebA_new' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MBGAN_celebA_new' --batch_size 11 --mode 'train' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --dataset1_crop_size 256

# python3.5 main.py  --task_name 'MBGAN_celebA_new' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --image_size 128
# python3.5 main.py  --task_name 'mbgan_add_sigmoid_makeup' --batch_size 1 --mode 'vis' --dataset2_image_path './data/makeup-data5.0-big-YZ-v2'

# python3.5 main.py  --task_name 'MBGAN_celebA_new' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --image_size 256

# python3.5 main.py  --task_name 'MBGAN_celebA_2_128' --batch_size 1 --mode 'evaluate' --dataset1_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/crop_images' --dataset 'CelebA' \
# --num_epochs 20 --num_epochs_decay 10 --log_step 1000 --sample_step 8000 --model_save_step 16600 --image_size 128

# python3.5 main.py  --task_name 'MBGAN_rafd' --batch_size 2 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/Rafd' --pretrained_model '' \
# --c_dim 8 --c2_dim 8 --model_save_step 7200 --sample_step 1000 \
# --dataset1_crop_size 680 --dataset2_crop_size 600 --image_size 600 --lambda_rec 10

python3.5 main.py  --task_name 'MBGAN_360_faceaging' --batch_size 11 --mode 'train' --dataset2_image_path '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/360_aging' --pretrained_model 'MBGAN_celebA/models/20_16600' \
--c_dim 7 --c2_dim 7 --model_save_step 1840 --sample_step 1840
