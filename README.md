#BUT-FIT at SemEval-2019 Task 7: Determining the Rumour Stance with Pre-Trained Deep Bidirectional Transformers

__Authors__:
* Martin Fajčík
* Lukáš Burget
* Pavel Smrž

This is a official implementation we have used in the SemEval-2019 Task 7. Our preview of submitted paper under review is available at https://arxiv.org/pdf/1902.10126v1.pdf   

## Bibtex citation
```
@article{fajcik2019but,
  title={BUT-FIT at SemEval-2019 Task 7: Determining the Rumour Stance with Pre-Trained Deep Bidirectional Transformers},
  author={Fajcik, Martin and Burget, Luk{\'a}{\v{s}} and Smrz, Pavel},
  journal={arXiv preprint arXiv:1902.10126},
  year={2019}
}
```


## Table of contents
- [Repository Description](#repository-description)
- [Replication of results](#replication-of-results)
  * [Replication from ensemble predictions](#replication-from-enseble-predictions)
  * [Replication via training new models](#replication-via-training-new-models)
- [Prediction examples](#prediction-examples)
  * [Structured self-attention](#structured-self-attention)
  * [BERT](#bert)
- [Visualisation](#visualisation)
  * [Attention from BERT - images](#attention-from-bert---images)
  * [Structured Self-attention with BERT-pretrained embeddings](#structured-self-attention-with-bert-pretrained-embeddings)

## Repository description
tbd.
## Replication of results
### Replication from ensemble predictions
Since each trained model is saved in checkpoints of size 1.3GB, we do not provide these online.
To replicate the ensemble results from paper, we provide a set of pre-calculated predictions frfom these trained models per validation set and per test set.
The predictions on validation and test sets are saved as numpy arrays in [predictions](predictions) folder.


### Replication via training new models
tbd.
## Prediction examples
### Structured self-attention
tbd.
### BERT
tbd.
## Visualisation
### Attention from BERT - images
The images of multi-head attention from all heads and layers are available for download [HERE](www.stud.fit.vutbr.cz/~ifajcik/example_attention.zip). 

### Structured self-attention with BERT-pretrained embeddings
tbd.