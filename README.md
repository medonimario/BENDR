# BENDR 
*BErt-like Neuro Data Representation*

This repository is a more user-friendly extension of the repository used to produce the results in the article:

[BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data](https://arxiv.org/pdf/2101.12037.pdf)

If you use this code, please cite the original article.

## BErt-like Neuro Data Representation (BENDR)
BERT-inspired Neural Data Representations (BENDR) is a method that applies successful language modeling techniques from fields like speech and image recognition to EEG data for improved brain-computer interface (BCI) classification and diagnosis. BENDR uses unlabeled data from different subjects, sessions, and tasks to learn the broader distribution of EEG data before fine-tuning on a specific task with limited labeled data.

## Repository
The original code for BENDR is implemented using the [DN3](https://dn3.readthedocs.io/en/latest/) which is a framework for deep learning in neuroscience built on PyTorch with little to no documentation. This repository is a more user-friendly extension of the original repository. This is done by making the model into a PyTorch class with familiar methods with familiar inputs and outputs.

### Pre-trained Model
In the original [paper](https://arxiv.org/pdf/2101.12037.pdf), the encoder and contextulizer of the model was pre-trained on [TUEG dataset](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml), which is a massive dataset with more than 2 TB of EEG data. For most purposes, it would be infeasible to pre-train the model on this dataset again, hence the pre-trained contextualizer weights are provided [here](https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/contextualizer.pt) (584 MB) and the pre-trained encoder weights are provided [here] (https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt) (15.2 MB).