
python preprocess_data.py --dataset dataset/ncbi-disease/ \
                          --train_data train_dev.json \
                          --max_ent_len 64

python generate_candidates.py --dataset dataset/ncbi-disease/ \
                          --model model_retriever/ncbi_retriever.pt \
                          --gpus 2
python evaluate.py --dataset dataset/ncbi-disease/ \
                    --model model_disambiguation/ncbi_disambiguation_prompt_pretrain.pt \
                    --pretrained_model_path model_pretrain/bc5cdr_pretrain.pt \
                    --gpus 2 \
                    --use_pretrained_model