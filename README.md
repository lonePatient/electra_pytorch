
## electra_pytorch

This repository contains a PyTorch implementation of the electra model from the paper 

[ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators ](https://openreview.net/pdf?id=r1xMH1BtvB)

by Kevin Clark. Minh-Thang Luong. Quoc V. Le. Christopher D. Manning

**NOTE**： 🤗This version is experience version,and the offical PyTorch version is waiting for the update of 🤗[huggingface](https://github.com/huggingface/transformers)

## Dependencies

- pytorch=1.10+
- cuda=9.0
- cudnn=7.5
- scikit-learn
- sentencepiece
- python3.6+

## Download Pre-trained Models 

**English**: Official download links: [google electra](https://github.com/google-research/electra)

**Chinese**: https://github.com/CLUEbenchmark/ELECTRA

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
    --electra_config_file=./prev_trained_model/electra_large/config.json \
    --pytorch_dump_path=./prev_trained_model/electra_large/pytorch_model.bin
```

Before running anyone of these GLUE/CLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) /[CLUE data](https://www.cluebenchmarks.com/introduce.html) by running  script named `download_xxxx_data` in the directory`tools` and unpack it to some directory $DATA_DIR.

3．run `sh run_classifier_sst2.sh`to fine tuning albert model

## Result

Performance of **electra** on GLUE benchmark results using a single-model setup on **dev**:

|  | Cola| Sst-2| Mnli| Sts-b|
| :------- | :---------: | :---------: |:---------: | :---------: |
| metrics | matthews_corrcoef | accuracy | accuracy | pearson |
| electra_small | 56.6 | 90.5 |  | 87.6 |
| electra_base | 67.8 | 94.2 |  | 91.1 |
| electra_large | 71.1 | 95.8 |  | 92.4 |

Performance of **electra** on CLUE benchmark results using a single-model setup on **dev**:


|  | AFQMC| TNEWS | IFLYTEK |
| :------- | :---------: | :---------: |:---------: |
| metrics | accuracy | accuracy | accuracy |
| electra_tiny | 69.82 | 54.48 | 56.98 |

## sample

Temperature $T  >0 $ is a hyper-parameter that regulates the probability distribution $p_i$ of the token. We divide the logits $z_i$ by $T$ before  computing  the `softmax`.

$$
p_i = \frac{\math{exp}(z_i / T)}{\sum_{j} \math{exp}(z_j / T)}
$$

$T= 1$ yields the unmodified distribution.

```python
def temperature_sampling(logits, temperature,do_sample=True):
    assert temperature >=0
    if do_sample:
        if temperature != 1.0:
            logits = logits / temperature
        # Sample
        batch_size,sequence_size,hidden_size = logits.size()
        logits = logits.view(-1,hidden_size)
        token_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        token_ids = token_ids.view(batch_size,sequence_size)
    else:
        # Greedy decoding
        token_ids = torch.argmax(logits, dim=-1)
    return token_ids
```

## pretraining

Small model on small dataset.

![](./outputs/training.png)
