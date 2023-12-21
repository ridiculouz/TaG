# TaG
The official repository for ACL 2023 paper "[A Novel Table-to-Graph Generation Approach for Document-Level Joint Entity and Relation Extraction](https://aclanthology.org/2023.acl-long.607)".

## Requirements

* Python (tested on 3.8.10)
* CUDA (tested on 11.2)
* [PyTorch](http://pytorch.org/) (tested on 1.12.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.12.4)
* [apex](https://github.com/NVIDIA/apex) (tested on 0.1)
* [opt-einsum](https://github.com/dgasmith/opt_einsum) (tested on 3.3.0)
* numpy
* scikit-learn
* scipy
* wandb
* ujson
* tqdm

## Dataset
The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions at [here](https://github.com/thunlp/DocRED/tree/master/data). 
The [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/) dataset is a revised version of DocRED and can be downloaded from [here](https://github.com/tonytan48/Re-DocRED).
The expected structure of files is:
```
TaG
 |-- dataset
 |    |-- docred/re-docred
 |    |    |-- train_annotated.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- rel_info.json
 |    |    |-- rel2id.json (in this repository)
```
We provide a copy of DocRED in `data/docred`. It is worth noting that TaG also need to generate intermediate data (`{}-gc.json`) from mention extraction prediction results, and we also provide the example files in `data/docred`.

## Training & Evaluation

Since we detach the mention extraction (ME) stage from the end-to-end pipeline, you can train a TaG model in the following steps:

1. Run mention extraction with `src/train_me.py`. An example script is `run_me.sh`.
2. Preprocess span prediction with `src/prepro.py`. Specify the dev and test predictions with `--dev_file` and `--test_file` arguments respectively.
3. Run coreference resolution & relation extraction with `src/train_tag.py`. An example script is `run_tag.sh`.

The evaluation results are provided in the log. To evaluate RE result on test data, you should first save the model using `--save_path` argument before training. The model correponds to the best dev results will be saved. After that, You can evaluate the saved model by setting the `--load_path` argument, and the program will generate a test file `result.json`.

## Acknowledgement

In this repository, we refer to and use some code from [ATLOP](https://github.com/wzhouad/ATLOP). Thanks for their open-source efforts!🍻

## Citation
```bibtex
@inproceedings{zhang-etal-2023-novel,
    title = "A Novel Table-to-Graph Generation Approach for Document-Level Joint Entity and Relation Extraction",
    author = "Zhang, Ruoyu and Li, Yanzeng and Zou, Lei",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.607",
    doi = "10.18653/v1/2023.acl-long.607",
    pages = "10853--10865",
}
```
