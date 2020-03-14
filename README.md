
## electra_pytorch

This repository contains a PyTorch implementation of the electra model from the paper 

[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators ](https://openreview.net/pdf?id=r1xMH1BtvB)

by Kevin Clark. Minh-Thang Luong. Quoc V. Le. Christopher D. Manning

## Dependencies

- pytorch=1.10+
- cuda=9.0
- cudnn=7.5
- scikit-learn
- sentencepiece
- python3.6+

## Download Pre-trained Models of English

Official download links: [google electra](https://github.com/google-research/electra)

## Fine-tuning

１. Place `config.json` into the `prev_trained_model/electra_base` directory.
example:
```text
├── prev_trained_model
|  └── electra_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
```
2．convert electra tf checkpoint to pytorch
```python
python convert_electra_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/electra_large \
    --bert_config_file=./prev_trained_model/electra_large/config.json \
    --pytorch_dump_path=./prev_trained_model/electra_large/pytorch_model.bin
```
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory $DATA_DIR.

3．run `sh run_classifier_sst2.sh`to fine tuning albert model

## Result

Performance of **electra** on GLUE benchmark results using a single-model setup on **dev**:


|  | Cola| Sst-2| Mnli| Sts-b|
| :------- | :---------: | :---------: |:---------: | :---------: |
| metric | matthews_corrcoef |accuracy |accuracy | pearson |

| model | Cola| Sst-2| Mnli| Sts-b|
| :------- | :---------: | :---------: |:---------: | :---------: |
| electra_small | 56.6 | 90.5 |  | 87.6 |
| electra_base | 67.8 | 94.2 |  | 91.1 |
| electra_large | 71.1 | 95.8 |  | 92.4 |

## pretraining

Small model on small dataset.

![](./outputs/training.png)
