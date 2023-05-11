
## Medical AI Research Foundations

[Medical AI Research Foundations](https://doi.org/10.13026/psq3-vj24) is a repository of open-source medical foundation models and code. Researchers and developers can accelerate their medical AI research by deploying this collection of non-diagnostic models, APIs, and resources . This is a clear unmet need, as currently there is no central resource that developers and researchers can leverage to build medical AI. This has slowed down both research and translation efforts. Our goal is to democratize access to foundational medical AI models and help researchers and medical AI developers rapidly build new solutions. To this end, we open-sourced [REMEDIS](https://arxiv.org/pdf/2205.09723.pdf) code-base and we are currently hosting checkpoints for chest x-ray and pathology at [PhysioNet](https://doi.org/10.13026/psq3-vj24). We expect to add more models and resources for training medical foundation models such as datasets and benchmarks in future. We also welcome the medical AI research community to contribute to this.



###   **Technical Implementation**

Overall, our approach comprises the following steps:

1. Supervised representation learning on a large-scale dataset of labeled natural images
2. Self-supervised contrastive representation learning on an unlabeled dataset of in-distribution medical images
3. Supervised fine-tuning on labeled in-distribution medical images

We open-source models that are the result of step 2. These models provide strong starting points for researchers developing diagnostic machine learning models.

The models use ResNet as the architecture backbone.

A brief description of the pre-training procedure follows below. For a full summary, please refer to the [REMEDIS article](https://arxiv.org/pdf/2205.09723.pdf) where we provide detailed descriptions of the preprocessing, pretraining, finetuning and hyperparameters for each of the tasks and models.

We begin with ResNet backbone models initialized with weights from [Big Transfer (BiT)](https://github.com/google-research/big_transfer) pretrained models. In addition to the model architecture, BiT models vary based on the pretraining dataset: BiT-S, BiT-M and BiT-L, where S(mall), M(edium) and L(arge) indicate if the pretraining was done on ILSVRC-2012 (ImageNet-1K), ImageNet-21K or JFT, respectively. We open source models based on BiT-M only.

For contrastive pretraining, we build on [SimCLR](https://github.com/google-research/simclr), which proposes a simple approach for contrastive learning for images. We performed a disjoint hyper-parameter tuning procedure to select factors influencing the quality of the learned representation, which we measured by the model performance in the downstream tasks using the validation set of the in-distribution data.

In our default contrastive pretraining setting, we utilized random cropping (C), random color distortion (D), rotation (R), and random Gaussian blur (G) as the data augmentation strategy. Due to the grayscale nature of radiology images, for these images we opted for stronger data augmentation to reduce the chances of overfitting. We further improved the final performance by incorporating histogram equalization and elastic deformation  in addition to our default data augmentation strategy.


### **Training Data**
We open-source models trained on public medical data only. This is available for chest x-ray and pathology only. The data used in each model are the following:

- Chest X-Ray
  - MIMIC-IV - CXR: This is a large, publicly available dataset of chest radiographs in JPG format. It is wholly derived from [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), with the JPG files derived from the DICOM images and the structured labels from free-text reports.
  - [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/): This is a large open source dataset of 224,316 de-identified CXRs from 65,240 unique patients. We specifically use the five most prevalent pathologies, including atelectasis, consolidation, pulmonary edema, pleural effusion, and cardiomegaly.

- Pathology
  - The Cancer Genome Atlas ([TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)): A random sample of 50M patches from 10,705 cases (29,018 slides) spanning 32 “studies” (cancer types) from TCGA is used.


### **Model Description**

All models comprise convolutional neural networks pre-trained with Big Transfer representation learning, and contrastively trained with SimCLR self-supervision. All models are a ResNet family model pre-trained with Big Transfer representation learning, and contrastively trained with SimCLR self-supervision.

We provide  ResNet 50x1 and Resnet 152x2 models for both the tasks. The models were pretrained at a resolution of 224x224 using Tensorflow and available as TF Hub weights. The suffix `-m` and `-s` refer to models pretrained using BiT-M and BiT-S respectively as the starting point.

For further details on datasets used to train the models and finetuning procedure, please refer to  [our paper](https://arxiv.org/pdf/2205.09723.pdf). The code can be found at colabs directory.


### **Installation and Requirements**

Models are provided as TensorFlow 2 saved models, and are compatible with versions of TF above 2.11. Beyond this, there are no requirements. To install tensorflow in your python runtime, please see the [TensorFlow documentation](https://www.tensorflow.org/install).


**Inference**: These models can be used as fixed embedding models that produce image representations to then train other models on. To only run inference, no complex hardware is needed. Simply load the model as shown, and perform inference.

**Finetuning**: These models can be used for full end-to-end fine-tuning on radiology or pathology data.  Although fine-tuning of these models could be done on any hardware, it will be slow. Simply loading the data alone on some hardware may be slow or impossible (the patch_camelyon dataset provided on tensorflow datasets is 7.48GiB in size, with ~330 thousand images). Hence GPU or TPU is suggested in these cases.


### **Files**
You can access model weights at the [Medical AI Research Foundations PhysioNet](https://physionet.org/content/medical-ai-research-foundation/1.0.0/) after acknowledging the usage license.


| Model                                                          | Modality         | Backbone | Architecture    |
|----------------------------------------------------------------|------------------|----------|-----------------|
| [cxr-152x2-remedis-m](https://doi.org/10.13026/grp0-z205)      | Chest X-Ray      | BiT-M    | ResNet 152x2    |
| [cxr-152x2-remedis-s](https://doi.org/10.13026/grp0-z205)      | Chest X-Ray      | BiT-S    | ResNet 152x2    |
| [cxr-50x1-remedis-m](https://doi.org/10.13026/grp0-z205)       | Chest X-Ray      | BiT-M    | ResNet 50x1     |
| [cxr-50x1-remedis-s](https://doi.org/10.13026/grp0-z205)       | Chest X-Ray      | BiT-S    | ResNet 50x1     |
| [path-152x2-remedis-m](https://doi.org/10.13026/grp0-z205)     | Pathology        | BiT-M    | ResNet 152x2    |
| [path-152x2-remedis-s](https://doi.org/10.13026/grp0-z205)     | Pathology        | BiT-S    | ResNet 152x2    |
| [path-50x1-remedis-m](https://doi.org/10.13026/grp0-z205)      | Pathology        | BiT-M    | ResNet 50x1     |
| [path-50x1-remedis-s](https://doi.org/10.13026/grp0-z205)      | Pathology        | BiT-S    | ResNet 50x1     |



There are multiple models provided. Each model file has the following format:

`{DATA_TYPE}-{ARCHITECTURE}-remedis-{PRETRAINING_DATA_SIZE}`

  - `DATA_TYPE`:  `cxr` (for Chest X-Ray) or `path` (for Pathology).
  - `ARCHITECTURE`: 50x1 (for ResNet 50x1) or 152x2 (for ResNet 152x2), indicating the architectures.
  - `RETRAINING_DATA_SIZE`: `s` or `m`, indicating whether BiT-S or BiT-M were used as a starting point.


Download the models using the terminal and the following command or by visiting [Medical AI Research Foundations PhysioNet](https://physionet.org/content/medical-ai-research-foundation/1.0.0/) directly:


```
wget -r -N -c -np --user <physionet-username> --ask-password https://physionet.org/files/medical-ai-research-foundation/1.0.0/
```


###  **Example Usage**
The Tensorflow 2 saved model format can be loaded as follows. See further information about hub Module [here](https://www.tensorflow.org/hub/api_docs/python/hub/Module).

```
import tensorflow_hub as hub

module = hub.load('TOP_LEVEL_HUB_PATH')

# Pathology: The image is of shape (<BATCH_SIZE>, 224, 224, 3)
# Chest X-Ray: The image is of shape (<BATCH_SIZE>, 448, 448, 3)
image = <LOAD_IMAGE_HERE>

embedding_of_image = module(image)
```


###  **Pretraining**

To pretrain the model on your dataset, you need to setup your data pipeline at [data.py](https://github.com/google-research/medical-ai-research-foundations/blob/main/data.py) this could be either a builder for TFDS dataset or any other format such as TFRecord.

Pipeline will take care of preprocessing and augmentation when builder is correctly defined and setup.  Here for example for a CIFAR-10 with a single GPU, try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False

```

### **Finetuning**

 For example usecase of fine-tuning of our checkpoint on CheXpert and Camelyon, please see our code at colabs directory.


### **Usage Notes**

We believe these models are best used for either full, end-to-end finetuning on radiology or pathology data, or as fixed embedding models that produce image representations to then train other models on. While we have attempted to rigorously evaluate our models in diverse tasks and settings, they may still fail when encountering data from unseen environments. Further, the impact of large scale self-supervised learning on fairness and safety is an open topic of research. We hope the release of these models will spur further research here.


### **Cite**

We kindly request that user cite the corresponding papers if you use our checkppoints or our code in any capacity. Proper attribution helps acknowledge and support the original authors' hard work.


```
@article{azizi2022robust,
  title={Robust and efficient medical imaging with self-supervision},
  author={Azizi, Shekoofeh and Culp, Laura and Freyberg, Jan and Mustafa, Basil and Baur, Sebastien and Kornblith, Simon and Chen, Ting and MacWilliams, Patricia and Mahdavi, S Sara and Wulczyn, Ellery and others},
  journal={arXiv preprint arXiv:2205.09723},
  year={2022}
}

@misc{azizi2023medical,
  author = {Azizi, S. and Freyberg, J. and Culp, L. and MacWilliams, P. and Mahdavi, S. and Natarajan, V. and Karthikesalingam, A.},
  title = {Medical AI Research Foundations: A repository of medical foundation models (version 1.0.0). PhysioNet.},
  url = {https://doi.org/10.13026/grp0-z205},
  year = {2023},
}
```


**This is not an officially supported Google product.**
