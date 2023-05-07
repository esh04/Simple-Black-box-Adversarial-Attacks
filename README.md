# Simple-Black-box-Adversarial-Attacks

Simple black box adversarial attacks refer to the methods of generating adversarial examples for machine learning models without any knowledge of the model's architecture or parameters. These attacks aim to find small perturbations to the input that can cause the model to make incorrect predictions.

## Implementation

- Finetuned the Resnet50 on the TinyImageNet dataset to now classify into 200 classes.
- Attacks implemented
    - Untargetted Attack
        - Cartesian Basis
        - Discrete Cosine Basis
    - Targetted Attack

## Repository Structure

Link to the paper refered: https://arxiv.org/abs/1905.07121

* [Simple-Black-box-Adversarial-Attacks](./Simple-Black-box-Adversarial-Attacks)
    * [README.md](./README.md)
    * [bashscripts](./bashscripts)
        * [attack.sh](./bashscripts/attack.sh)
        * [black-box.sh](./bashscripts/black-box.sh)
    * [data](./data)
        * [README.md](./data/README.md)
    * [logs](./logs)
        * [slurm](./logs/slurm)
    * [model](./model)
        * [README.md](./model/README.md)
    * [pyscripts](./pyscripts)
        * [black-box.py](./pyscripts/black-box.py)
        * [targetted.py](./pyscripts/targetted.py)
        * [untargetted.py](./pyscripts/untargetted.py)
        * [untargetted_dct.py](./pyscripts/untargetted_dct.py)
    * [Presentation.pdf](./Presentation.pdf)
    * [requirements.txt](./requirements.txt)
    * [graphs.ipynb](./graphs.ipynb)
    * [Adverse_Attack_Examples.ipynb](./Adverse_Attack_Examples.ipynb)


- Due to lack of space all the data is uploaded to onedrive, links of which can be found in the README.md of the data folder. The same is applicable for the model

- bash_scripts contains the scripts to run the models for various values of e.

- logs contains the training log for finetuning of Resnet50 on TinyImageNet

- pyscripts contains the code for every attack
    - Untargetted SimBA
    - Untargetted SimBA-DCT
    - Targetted SimBA
    and also the code for finetuning the resnet50.

- graphs.ipynb contains the analysis and comparisons of all the methods used.
- Adverse_Attack_Examples.ipynb contains the examples of all the methods.
