# Prompt-BioEL
<p align="center">
  <img src="pic/re-ranker-v5.png" width="650"/>
</p>

An entity re-ranking model based on **prompt tuning** for **biomedical entity linking**,
along with a KB-enhanced self-supervised pretraining strategy.
More details can be found in our paper
"Improving Biomedical Entity Linking with Cross-Entity Interaction" (accepted by [AAAI 2023](https://aaai-23.aaai.org/)).


## 🚨: Usage

### Environment
```
conda activate -n bioEL python=3.9
conda activate bioEL
pip install -r requirements.txt
```

### Data and Checkpoints
Please see the `README.md` files in different folders to download the corresponding data and checkpoints.

### Evaluate with Our Checkpoints
After downloading the data and checkpoints, you can use the command below to replicate our results reported in the paper.
If you want to train your own model, please skip to [data preprocessing step](#Preprocess-Data).
- NCBI-Disease
```
bash eval_ncbi.sh
```
- BC5CDR
```
bash eval_bc5cdr.sh
```
- COMETA
```
bash eval_cometa.sh
```

### Preprocess Data
You can use the command below the to prepare the data for training the retriever.
- NCBI-Disease
```
python preprocess_data.py --dataset dataset/ncbi-disease/ \
                          --train_data train_dev.json \
                          --max_ent_len 64
```

- BC5CDR
```
python preprocess_data.py --dataset dataset/bc5cdr/ \
                          --train_data train.json \
                          --max_ent_len 128
```
- COMETA
```
python preprocess_data.py --dataset dataset/cometa/ \
                          --train_data train.json \
                          --max_ent_len 64
```
### Train Retriever
After the preparation, you can train the retriever with the command below.
- NCBI-Disease
```
python run_retriever.py --dataset dataset/ncbi-disease/ \
                        --model model_retriever/ncbi_retriever.pt \
                        --epochs 17 \
                        --gpus 0
```

- BC5CDR
```
python run_retriever.py --dataset dataset/bc5cdr/ \
                        --model model_retriever/bc5cdr_retriever.pt \
                        --epochs 20 \
                        --gpus 0
```
- COMETA
```
python run_retriever.py --dataset dataset/cometa/ \
                        --model model_retriever/cometa_retriever.pt \
                        --epochs 20 \
                        --gpus 0
```

### Pretrain
To improve the reranking performance, you can pretrain the model with the corresponding knowledge base\(KB\). If you want to train the model directly, 
please skip to the [reranker training step](#Train-Reranker).

- BC5CDR
```
python run_pretrain.py --dataset dataset/bc5cdr/ \
                      --model model_pretrain/bc5cdr_pretrain.pt \
                      --epochs 15 \
                      --gpus 0
```
- COMETA
```
python run_pretrain.py --dataset dataset/cometa/ \
                      --model model_pretrain/cometa_pretrain.pt \
                      --epochs 10 \
                      --gpus 0
```

### Train Reranker
After retrieving the candidate entities, you can train the reranker with the command below to get the final answer.
If you do not pretrain the model or use our checkpoint, either, the `--use_pretrained_model` is not needed anymore.
- NCBI-Disease
```
python run_disambiguation_prompt.py --dataset dataset/ncbi-disease/ \
                                    --model model_disambiguation/ncbi_disambiguation_prompt_pretrain.pt \
                                    --pretrained_model_path model_pretrain/bc5cdr_pretrain.pt \
                                    --epochs 9 \
                                    --gpus 1 \
                                    --use_pretrained_model
```

- BC5CDR
```
python run_disambiguation_prompt.py --dataset dataset/bc5cdr/ \
                                    --model model_disambiguation/bc5cdr_disambiguation_prompt_pretrain.pt \
                                    --pretrained_model_path model_pretrain/bc5cdr_pretrain.pt \
                                    --epochs 28 \
                                    --gpus 0 \
                                    --use_pretrained_model
```
- COMETA
```
python run_disambiguation_prompt.py --dataset dataset/cometa/ \
                                    --model model_disambiguation/cometa_disambiguation_prompt_pretrain.pt \
                                    --pretrained_model_path model_pretrain/cometa_pretrain.pt \
                                    --epochs 40 \
                                    --gpus 0 \
                                    --use_pretrained_model
```


