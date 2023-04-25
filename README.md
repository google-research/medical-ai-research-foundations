
## Medical AI Research Foundations

Recent progress in Medical AI has resulted in systems reaching clinical expert level performance, but they struggle with sub-optimal performance in different clinical settings. Developing separate systems for each setting is impractical due to time and cost constraints of acquiring and annotating medical data. [REMEDIS](https://arxiv.org/pdf/2205.09723.pdf), a unified representation learning strategy, addresses this challenge by combining supervised transfer learning with self-supervised learning and requiring minimal task-specific customization. Results of retrospective study show that REMEDIS improves in-distribution performance with up to 11.5% relative improvement in diagnostic accuracy, and achieves strong data-efficient generalization, matching supervised baselines with as little as 1% to 33% of retraining data across tasks. REMEDIS has the potential to accelerate medical imaging AI development and deliver broad impact.

The goal of our method is to learn a predictor for each domain-specific medical task with low prediction error on both the in-distribution and the out-of-distribution data. Since it has been shown pretraining on massive unlabeled datasets potentially improves accuracy under distribution shift, here we focused on predictors that leverage these pretrained representations and further fine-tuned using the labeled data. In the representation learning step, we trained an encoder f(·) to produce representations by minimizing some loss; cross-entropy loss (a multi-class generalization of logistic loss) for supervised pretraining or a contrastive loss for self-supervised pretraining.

Please refer to [our paper for experimental results](https://arxiv.org/pdf/2205.09723.pdf). Overall, the models we trained greatly reduce the need for supervised learning data and can thus serve as strong foundations for researchers building medical imaging models for chest x-ray and pathology. We are looking forward to feedback from the community to help improve our models. In particular, we hope the release of these models will spur research into open questions such as the impact of large scale supervision on aspects such as fairness, bias, safety and privacy.


#### **Model Description**

We open-source models trained on public medical data only. This is available for chest x-ray and pathology only. The data supporting our models therefore is:

- Chest X-Ray
  - MIMIC-IV
  - CheXpert
- Pathology
  - Camelyon-16
  - Camelyon-17

All models comprise convolutional neural networks pre-trained with Big Transfer representation learning, and contrastively trained with SimCLR self-supervision. All models are a ResNet family model pre-trained with Big Transfer representation learning, and contrastively trained with SimCLR self-supervision.

We provide  ResNet 50x1 and Resnet 152x2 models for both the tasks. The models were pretrained at a resolution of 224x224 using Tensorflow and available as TF Hub weights. The suffix `-m` and `-s` refer to models pretrained using BiT-M and BiT-S respectively as the starting point.

For further details on datasets used to train the models and finetuning procedure, please refer to  [our paper](https://arxiv.org/pdf/2205.09723.pdf). The code can be found at colabs directory.

####   **Technical Implementation**

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

#### **Installation and Requirements**

Models are provided as TensorFlow 2 saved models, and are compatible with versions of TF above 2.11. Beyond this, there are no requirements. To install tensorflow in your python runtime, please see the [TensorFlow documentation](https://www.tensorflow.org/install).

####  **Pretraining**

To pretrain the model on your dataset, you need to setup your data pipeline at [data.py](https://github.com/google-research/medical-ai-research-foundations/blob/main/data.py) this could be either a builder for TFDS dataset or any other format such as SStable or TFRecord format.

Pipeline will take care of preprocessing and augmentation when builder is correctly defined and setup.  Here for example for a CIFAR-10 with a single GPU, try the following command:

```
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/tmp/simclr_test --use_tpu=False

```

#### **Finetuning**

For example usecase of fine-tuning of our checkpoint on CheXpert and Camelyon, please see our code at colabs directory.


#### **Usage Notes**

We believe these models are best used for either full, end-to-end finetuning on radiology or pathology data, or as fixed embedding models that produce image representations to then train other models on. While we have attempted to rigorously evaluate our models in diverse tasks and settings, they may still fail when encountering data from unseen environments. Further, the impact of large scale self-supervised learning on fairness and safety is an open topic of research. We hope the release of these models will spur further research here.


#### **Cite**

We kindly request that user cite the corresponding papers if you use our checkppoints or our code in any capacity. Proper attribution helps acknowledge and support the original authors' hard work.


```
@article{azizi2022robust,
  title={Robust and efficient medical imaging with self-supervision},
  author={Azizi, Shekoofeh and Culp, Laura and Freyberg, Jan and Mustafa, Basil and Baur, Sebastien and Kornblith, Simon and Chen, Ting and MacWilliams, Patricia and Mahdavi, S Sara and Wulczyn, Ellery and others},
  journal={arXiv preprint arXiv:2205.09723},
  year={2022}
}

@inproceedings{azizi2021big,
  title={Big self-supervised models advance medical image classification},
  author={Azizi, Shekoofeh and Mustafa, Basil and Ryan, Fiona and Beaver, Zachary and Freyberg, Jan and Deaton, Jonathan and Loh, Aaron and Karthikesalingam, Alan and Kornblith, Simon and Chen, Ting and others},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3478--3488},
  year={2021}
}

```


**This is not an officially supported Google product.**
