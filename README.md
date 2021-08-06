# Latent Space Regularization for Unsupervised Domain Adaptation in Semantic Segmentation
This is the official PyTorch implementation of our works: "Latent Space Regularization for Unsupervised Domain Adaptation in Semantic Segmentation" presented at the CVPR2021 Workshop on Autonomous Driving and "Adapting Segmentation Networks to New Domains by Disentangling Latent Representations" submitted to IEEE Transactions on Multimedia.

In this paper we propose an effective Unsupervised Domain Adaptation (UDA) strategy, based on feature clustering, perpendicularity and norm alignment objectives to reach a regularized disposition of latent embeddings, while providing a semantically consistent domain alignment over the feature space.

We evaluate our framework in synthetic-to-real and real-to-real scenarios, in particular with GTA5, SYNTHIA, Cityscapes and Cross-City datasets.

The webpage of the paper is: [here](https://lttm.dei.unipd.it/paper_data/LSR/).

The CVPRW paper can be found on CVF'S [Open Access](https://openaccess.thecvf.com/content/CVPR2021W/WAD/html/Barbato_Latent_Space_Regularization_for_Unsupervised_Domain_Adaptation_in_Semantic_Segmentation_CVPRW_2021_paper.html).

![teaser](architecture.png)


## Requirements
This repository uses the following libraries:
- python (3.9.5)
- pytorch (1.9.0)
- torchvision (0.10.0)
- torchaudio (0.9.0)
- torchsummary (1.5.1)
- tensorboardX (2.4)
- tqdm (4.61.2)
- imageio (2.9.0)
- numpy (1.20.3)
- pillow (8.3.1)
- opencv-python (4.5.3.56)

A guide to install Pytorch can be found [here](https://pytorch.org/).

For an easier installation of the other libraries, we provide you the requirement file to be run
```python
pip install -r requirements.txt
```

## Setup

#### GTA5
Download the [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset.
The dataset splits (train.txt and val.txt) are provided in the datasets/splits/GTA5 folder.

#### Cityscapes
Download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.
The dataset splits (train.txt, val.txt, trainplus.txt, test.txt) are provided in the datasets/splits/Cityscapes folder.

#### SYNTHIA  
Download the [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/downloads/) dataset.
The dataset splits (train.txt, val.txt) are provided in the datasets/splits/SYNTHIA folder.

#### Cross-City
Download the [NTHU](https://yihsinchen.github.io/segmentation_adaptation_dataset/) dataset.
The dataset splits (train.txt, val.txt) for each city (Rome, Rio, Taipei, Tokyo) are provided in the datasets/splits/NTHU folder.

Such datasets and splits must be pointed to using the following command line arguments:
> --data_root_path (path to dataset) --list_path (path to split lists)
> --target_root_path (path to target dataset) --target_list_path (path to split lists)

#### (Optional) PRETRAINED BACKBONES

Download the official pytorch checkpoints pretrained on imagenet for the backbones you wish to use and place them into the pretrained_model folder:
- [ResNet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
-	[ResNet50](https://download.pytorch.org/models/resnet50-19c8e357.pth)
-	[VGG16](https://download.pytorch.org/models/vgg16-397923af.pth)
-	[VGG13](https://download.pytorch.org/models/vgg13-c768596a.pth)

## Training

### Source Pre-training (example)

GTA5 with ResNet101:

If you haven't already done so, download the ResNet101's weights pretrained on ImageNet from [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) and run:
> python tools/train_source.py --backbone "resnet101" --dataset "cityscapes" --checkpoint_dir "./log/gta5_rn101_source" --data_root_path (path to dataset) --list_path (path to split lists) --target_root_path (path to dataset) --target_list_path (path to split lists) --target_base_size "1280,640" --target_crop_size "1280,640" --num_classes 19

Alternatively, you can download the GTA5 pretrained weights from [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/EW6E0PzjIetJjGbFHXvxIjMBsY3xyeLc7o4vGyPd34Ry2w?e=RQROac).

### Unsupervised Domain Adaptation

- GTA5-to-Cityscapes with ResNet101:
   > python tools/train_UDA.py --backbone "resnet101" --pretrained_ckpt_file "gta5_rn101_source.pth" --source_dataset "gta5" --checkpoint_dir "./log/gta2city_rn101" --data_root_path (path to dataset) --list_path (path to split lists) --target_root_path (path to dataset) --target_list_path (path to split lists) --num_classes 19 --lambda_entropy .1 --IW_ratio .2 --lambda_norm 0.025 --lambda_ortho 0.2 --lambda_cluster 0.1

   A checkpoint of the adapted model can be found [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/EbbNl4ohIN1DplojAiHWlx4B-X1_3hFtGsjLGhIWaMJxQA?e=ZzhRNe).

- GTA5-to-Cityscapes (CS-full) with ResNet101:
   > python tools/train_UDA.py --backbone "resnet101" --pretrained_ckpt_file "gta5_rn101_source.pth" --source_dataset "gta5" --checkpoint_dir "./log/gta2cityfull_rn101" --data_root_path (path to dataset) --list_path (path to split lists) --target_root_path (path to dataset) --target_list_path (path to split lists) --num_classes 19 --lambda_entropy .1 --IW_ratio .2 --lambda_norm 0.025 --lambda_ortho 0.2 --lambda_cluster 0.1 --target_train_split "train_plus"

  A checkpoint of the adapted model can be found [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/EfneqoMg4M5PhuxLKWfpj98BnnoGPeNxV_NGNZ2rYdYrDA?e=L0cKcP).

- GTA5-to-Cityscapes with VGG-16:
   > python tools/train_UDA.py --backbone "vgg16" --pretrained_ckpt_file "gta5_vgg16_source.pth" --source_dataset "gta5" --checkpoint_dir "./log/gta2city_vgg16" --data_root_path (path to dataset) --list_path (path to split lists) --target_root_path (path to dataset) --target_list_path (path to split lists) --num_classes 19 --lambda_entropy .1 --IW_ratio .2 --lambda_norm 0.01 --lambda_ortho 0.4 --lambda_cluster 0.2

   A checkpoint of the adapted model can be found [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/EWHjy0cDn75Li27BOAEFJf4BDv6yuh88ngLxoMY_8gDZ7A?e=PD7jfw).

- SYNTHIA-to-Cityscapes with ResNet-101:
   > python tools/train_UDA.py --backbone "resnet101" --pretrained_ckpt_file "synthia_rn101_source.pth" --source_dataset "synthia" --checkpoint_dir "./log/synthia2city_rn101" --data_root_path (path to dataset) --list_path (path to split lists) --target_root_path (path to dataset) --target_list_path (path to split lists) --num_classes 16 --lambda_entropy .1 --IW_ratio 0 --lambda_norm 0.1 --lambda_ortho 0.2 --lambda_cluster 0.05

   A checkpoint of the adapted model can be found [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/EUN47owR9yhFk0-JygrwdcUB4-dc9Eaq1qr1j5s9i_KH7Q?e=m38fZq).

- Cityscapes-to-CrossCity with ResNet-101:
  > python tools/train_UDA.py --backbone "resnet101" --dataset "city" --pretrained_ckpt_file "city_rn101_source.pth" --source_dataset "cityscapes" --checkpoint_dir "./log/city2cc_rn101" --data_root_path (path to dataset) --list_path (path to split lists) --base_size "1280,640"  --target_root_path (path to dataset) --target_list_path path to split lists) --num_classes 13 --lambda_entropy .1 --IW_ratio .2 --lambda_norm 0.025 --lambda_ortho 0.2 --lambda_cluster 0.1

  A checkpoint of the model adapted to Rome can be found [here](https://unipdit-my.sharepoint.com/:u:/g/personal/francesco_barbato_unipd_it/Ec3Ey8f-qipMgyyOOSfWlpEB2oAUnpX-swdIpHcUDaL0iw?e=3uex8c).

### Evaluation (example)

GTA5-to-Cityscapes with ResNet101:
> python tools/evaluate.py --backbone "resnet101" --source_dataset "gta5" --checkpoint_dir "./eval/gta2city_rn101" --pretrained_ckpt_file "./log/gta2city_rn101/gta52cityscapesbest.pth" --data_root_path (path to dataset) --list_path (path to dataset) --num_classes 19 --batch_size 1 --split "test" --data_loader_workers 4


## Cite us
If you use this repository, please consider to cite us

    @InProceedings{barbato2021latent,
    author    = {Barbato, Francesco and Toldo, Marco and Michieli, Umberto and Zanuttigh, Pietro},
    title     = {Latent Space Regularization for Unsupervised Domain Adaptation in Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {2835-2845}
    }

## Acknowledgments
The structure of this code is largely based on [this](https://github.com/ZJULearning/MaxSquareLoss) and [this](https://github.com/LTTM/UDAclustering) repo.


## License

[Apache License 2.0](LICENSE)
