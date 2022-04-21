# DeLiCaTe

Implementation of the Paper "Chemical transformer Compression for accelerating both training and inference of molecular modeling" by Yi Yu and Karl Börjesson. We assumed that our strategy will accelerate both training and inference of chemical transformer for molecular modeling. We have now pulished our results in pre-print and will make the models available when the paper accepted.

<img src="example/TOC.png" width="100%" height="100%">

## Installing
The compression methods in this package is based heavily on the MolBERT from BenevolentAI. The link of MolBERT is shown below:

https://github.com/BenevolentAI/MolBERT

### Prerequisites
'''
python 3.7
numpy
rdkit 2019.03.1.0
scikit-learn 0.21.3
pytorch-lightning 0.8.4
transformers 3.5.1
pytorch 1.7.0
'''
### Install via Anaconda (recommended way)
```bash
git clone https://github.com/YiYuDL/DeLiCaTe.git
cd DeLiCaTe
conda create -y -q -n delicate -c rdkit rdkit=2019.03.1.0 python=3.7.3
conda activate delicate
pip install .
```
## Getting start
The compression methods here include cross-layer parameter sharing (CLPS), knowledge distillation (KD) and the integration of two mentioned methods. The obtained transformer models are PSMolBERT, KDMolBERT and DeLiCaTe, respectively. The model compression will be shown in turn. Then, the fine-tuning for QSAR and comparison of inference speed are shown. Finally, the inference speed among different transfomer models are compared.
### Cross-layer parameter sharing (CLPS)
You can use the guacamol dataset for CLPS pre-training as well as KD (links at the [bottom](https://github.com/BenevolentAI/MolBERT#data))
