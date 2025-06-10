for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method adaptformer --dim 8 --bit 32 --load_config --eval --model_path './result/adaptformer'
    done

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method adaptformer --r_layer 6 --dim 8 --bit 32 --load_config --eval --model_path './result/adaptformer-fpet'
    done


for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method lora --dim 8 --bit 32 --load_config --eval --model_path './result/lora'
    done

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method lora --r_layer 6 --dim 8 --bit 32 --load_config --eval --model_path './result/lora-fpet'
    done


for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method adaptformer --dim 32 --bit 1 --load_config --eval --model_path './result/bi-adaptformer'
    done

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method adaptformer --r_layer 6 --dim 32 --bit 1 --load_config --eval --model_path './result/bi-adaptformer-fpet'
    done
    

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method lora --dim 32 --bit 1 --load_config --eval --model_path './result/bi-lora'
    done

for DATASET in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --method lora --r_layer 6 --dim 32 --bit 1 --load_config --eval --model_path './result/bi-lora-fpet'
    done

