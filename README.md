# Simple-Black-box-Adversarial-Attacks

Simple black box adversarial attacks refer to the methods of generating adversarial examples for machine learning models without any knowledge of the model's architecture or parameters. These attacks aim to find small perturbations to the input that can cause the model to make incorrect predictions.

Link to the paper refered: https://arxiv.org/abs/1905.07121

Simple-Black-box-Adversarial-Attacks/
├── README.md
├── data/
│   └── README.md
├── bashscripts/
│   ├── black-box.sh
│   └── attack.sh
├── logs/
│   └── slurm
├── model/
│   └── README.md
├── pyscripts/
│   ├── targetted.py
│   ├── untargetted_dct.py
│   ├── untargetted.py
│   ├── black-box.py
├── graphs.ipynb
├── Adverse_Attack.ipynb
└── Presentation.pdf

- Due to lack of space all the data is uploaded to onedrive, links of which can be found in the README.md of the data folder. The same is applicable for the model

- bash_scripts contains the scripts to run the models for abalation study.

- logs contains the training log for finetuning of Resnet50 on TinyImageNet

- pyscripts contains the code for every attack
    - Untargetted SimBA
    - Untargetted SimBA-DCT
    - Targetted SimBA
    and also the code for finetuning the resnet50.

- graphs.ipynb contains the analysis
- adverse_affect.ipynb contains the examples
- Presentation made for the project
