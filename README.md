# BUT-FIT at SemEval-2019 Task 7: Determining the Rumour Stance with Pre-Trained Deep Bidirectional Transformers

__Authors__:
* Martin Fajčík
* Lukáš Burget
* Pavel Smrž

This is a official implementation we have used in the SemEval-2019 Task 7. Our preview of submitted paper under review is available at [arXiv](https://arxiv.org/pdf/1902.10126v1.pdf).
All models have been trained on RTX 2080 Ti (with 12 GB memory).

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
- [Replication of results](#replication-of-results)
  * [Replication from ensemble predictions](#replication-from-enseble-predictions)
  * [Replication via training new models](#replication-via-training-new-models)
- [Prediction examples](#prediction-examples)
  * [Structured self-attention](#structured-self-attention)
  * [BERT](#bert)
- [Visualisation](#visualisation)
  * [Attention from BERT - images](#attention-from-bert---images)
  * [Structured Self-attention with BERT-pretrained embeddings](#structured-self-attention-with-bert-pretrained-embeddings)

## Replication of results
### Replication from ensemble predictions
Since each trained model is saved in checkpoints of size 1.3GB, we do not provide these online.
To replicate the ensemble results from paper, we provide a set of pre-calculated predictions from these trained models per validation set and per test set.
The predictions on validation and test sets are saved as numpy arrays in [predictions](predictions) folder.

Running [replicate_ensemble_results.py](replicate_ensemble_results.py) directly replicates ensemble results.

### Replication via training new models
1. Make sure the value of `"active_model"` in [configurations/config.json](configurations/config.json) is set to `"BERT_textonly"`
2. Run solver.py  

Note: Mind that BERT often gets stuck in local minima. In our experiments, we took only results with 55 F1 on validation data or better.
For the sake of convenience, you may want to modify last line of method `create_model` found in  [solutionsA.py](solutionsA.py) file to call 
 `modelframework.fit_multiple` instead of `modelframework.fit` to run model training multiple times.

Duration of 1 training: ~ 30 minutes

### Replication of BiLSTM+SelfAtt baseline result
1. Change value of `"active_model"` in [configurations/config.json](configurations/config.json) to `"self_att_with_bert_tokenizer"`
2. Run solver.py

Duration of 1 training: ~ 2.7 minutes

## Prediction examples
### Structured self-attention with BERT-pretrained embeddings (BiLSTM+SelfAtt)
`tsv` file containing predictions, ground truth, confidence and model inputs of trained `BiLSTM+SelfAtt` model is available [HERE](https://www.stud.fit.vutbr.cz/~ifajcik/introspection_task_A.frameworks.self_att_with_bert_tokenizing.SelfAtt_BertTokenizing_Framework_F1_0.472417_LOSS_1.019169.tsv).
### BERT
`tsv` file containing predictions, ground truth, confidence and model inputs of trained `TOP-N_s` ensemble (our best published result) is available [HERE](https://www.stud.fit.vutbr.cz/~ifajcik/ensemble_introspection_TOP_N_s.tsv).
## Visualisation
### Attention from BERT - images
The images of multi-head attention from all heads and layers from trained BERT model for a fixed data point are available for download [HERE](https://www.stud.fit.vutbr.cz/~ifajcik/example_attention.zip). 

### Attention in structured self-attention with BERT-pretrained embeddings (BiLSTM+SelfAtt)
`xlsx` file containing attention visualisation per each input of validation set in trained `BiLSTM+SelfAtt` model is available [HERE](https://www.stud.fit.vutbr.cz/~ifajcik/introspection_task_A.frameworks.self_att_with_bert_tokenizing.SelfAtt_BertTokenizing_Framework_F1_0.472417_LOSS_1.019169.xlsx). 
The column description is shown in its first row.
For each example, column `'text'` contains numerical values of attention and visualisations of average over all attention "heads" and attention of each "head" (in this row order). Note, that at time attention is made, the input is already passed via 1-layer BiLSTM (see [original paper](https://arxiv.org/abs/1703.03130) for more details).
