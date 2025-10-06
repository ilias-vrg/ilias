#/bin/bash

ILIAS_ROOT=/path/to/images
YFCC_ROOT=/path/to/tars
FEATURE_ROOT=/path/to/hdf5

# AlexNet
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" tv alexnet gem 384 false
# VGG
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vgg16.tv_in1k gem 384 false
# ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm resnet50.tv_in1k head 384 false
# ResNet101
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm resnet101.tv_in1k head 384 false
# DenseNet169
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm densenet169.tv_in1k head 384 false
# InceptionV4
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm inception_v4.tf_in1k head 512 false
# NasNet-Large
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm nasnetalarge.tf_in1k head 512 false
# EfficientNet-B4
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm tf_efficientnet_b4.ns_jft_in1k head 512 false
# ViT-B/16 imagenet1k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_224.augreg_in1k head 384 false
# ViT-B/16 imagenet21k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_224.augreg_in21k head 384 false
# ViT-L/16 imagenet21k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_224.augreg_in21k head 384 false
# ViT-L/16 imagenet21k - ft imagenet1k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_224.augreg_in21k_ft_in1k head 384 false
# ViT-L/16@384 imagenet21k - ft imagenet1k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_384.augreg_in21k_ft_in1k head 512 false
# DeiT3 III base imagenet1k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm deit3_base_patch16_224.fb_in1k head 384 false
# DeiT3 III large imagenet1k
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm deit3_large_patch16_224.fb_in1k head 384 false
# OAI-CLIP ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" och RN50.openai head 384 true
# OAI-CLIP ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_clip_224.openai head 384 true
# OAI-CLIP ViT-L/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch14_clip_224.openai head 384 true
# OAI-CLIP ViT-L/14@336
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch14_clip_336.openai head 512 true
# OpenCLIP ViT-L/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch14_clip_224.laion2b head 384 true
# DINO ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dino_resnet50 head 384 false
# DINO ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dino_vitb16 head 384 false
# SWAV ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb swav head 384 false
# MoCo v3 ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb mocov3_resnet50 head 384 false
# MoCo v3 ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb mocov3_vitb16 head 384 false
# ConvNeXt-Base In1K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_base.fb_in1k head 384 false
# ConvNeXt-Large In1K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_large.fb_in1k head 384 false
# ConvNeXt-Base In22K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_base.fb_in22k head 384 false
# ConvNeXt-Large In22K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_large.fb_in22k head 384 false
# OpenCLIP ConvNeXt-Base
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_base.clip_laion2b_augreg head 384 true
# OpenCLIP ConvNeXt-Large@320
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm convnext_large_mlp.clip_laion2b_ft_soup_320 head 512 true
# RS@k ResNet50 SOP
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm recall_512-resnet50 head 384 false
# RS@k ViT-B/16 SOP
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm recall_512-vit_base_patch16_224_in21k head 384 false
# CVNet ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" cvn cvnet_resnet50 head 724 false
# CVNet ResNet101
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" cvn cvnet_resnet101 head 724 false
# SuperGlobal ResNet50
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" cvn superglobal_resnet50 head 724 false
# SuperGlobal ResNet101
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" cvn superglobal_resnet101 head 724 false
# Hier DINO ViT-B/16 SOP
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" hier dino_vit16_patch16_224_sop head 384 false
# EVA-02-B MIM In22K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm eva02_base_patch14_224.mim_in22k head 384 false
# EVA-02-L MIM In22K
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm eva02_large_patch14_224.mim_in22k head 384 false
# EVA-02-L MIM Merged-38M
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm eva02_large_patch14_224.mim_m38m head 384 false
# EVA-02-CLIP-B/16 Merged-2B
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm eva02_base_patch16_clip_224.merged2b head 384 true
# EVA-02-CLIP-L/14 Merged-2B
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm eva02_large_patch14_clip_336.merged2b head 512 true
# UNICOM ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unicom unicom_vit_base_patch16_224 head 384 false
# UNICOM ViT-L/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unicom unicom_vit_large_patch14_224 head 384 false
# UNICOM ViT-L/14@336
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unicom unicom_vit_large_patch14_336 head 512 false
# UNICOM ViT-B/16@512 GLDv2
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unicom unicom_vit_base_patch16_gldv2 head 724 false
# UNICOM ViT-B/16 SOP
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unicom unicom_vit_base_patch16_sop head 384 false
# USCRR ViT-B/16 - OAI-CLIP ViT-B/16 - ft UnED
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm uscrr_64-vit_base_patch16_clip_224.openai cls 384 false
# DINOv2 ViT-B/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov2_vitb14 head 724 false
# DINOv2 ViT-L/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov2_vitl14 head 724 false
# SigLIP ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_224.webli head 384 true
# SigLIP ViT-B/16@256
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_256.webli head 384 true
# SigLIP ViT-B/16@384
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_384.webli head 512 true
# SigLIP ViT-B/16@512
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_512.webli head 724 true
# SigLIP ViT-L/16@256
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_siglip_256.webli head 384 true
# SigLIP ViT-L/16@384
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_siglip_384.webli head 512 true
# MetaCLIP ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_clip_224.metaclip_2pt5b head 384 true
# MetaCLIP ViT-L/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch14_clip_224.metaclip_2pt5b head 384 true
# DINOv2 ViT-B/14-reg
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov2_vitb14_reg head 724 false
# DINOv2 ViT-L/14-reg
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov2_vitl14_reg head 724 false
# UNIC-L
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" unc unic_l head 512 false
# UDON ViT-B/16 - ViT-B/16 imagenet21k - ft UnED
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm udon_64-vit_base_patch16_224.augreg_in21k_ft_in1k cls 384 false
# UDON ViT-B/16 - OAI-CLIP ViT-B/16 - ft UnED
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm udon_64-vit_base_patch16_clip_224.openai cls 384 false
# SigLIP2 ViT-B/16@384
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_384.v2_webli head 512 true
# SigLIP2 ViT-B/16@512
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_base_patch16_siglip_512.v2_webli head 724 true
# SigLIP2 ViT-L/16@384
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_siglip_384.v2_webli head 512 true
# SigLIP2 ViT-L/16@512
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_large_patch16_siglip_512.v2_webli head 724 true
# PE ViT-B/14
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_pe_core_base_patch16_224.fb head 384 true
# PE ViT-L/14@336
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" timm vit_pe_core_large_patch14_336.fb head 512 true
# DINOv3 ViT-B/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov3_vitb16 head 768 false
# DINOv3 ViT-L/16
sbatch scripts/run_slurm.sh "$ILIAS_ROOT" "$YFCC_ROOT" "$FEATURE_ROOT" fb dinov3_vitl16 head 768 false