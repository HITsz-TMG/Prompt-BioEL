python preprocess_data.py --dataset dataset/bc5cdr/ \
                          --train_data train.json \
                          --max_ent_len 128

python generate_candidates.py --dataset dataset/bc5cdr/ \
                          --model model_retriever/bc5cdr_retriever.pt \
                          --gpus 2
python evaluate.py --dataset dataset/bc5cdr/ \
                    --model model_disambiguation/bc5cdr_disambiguation_prompt_pretrain.pt \
                    --pretrained_model_path model_pretrain/bc5cdr_pretrain.pt \
                    --gpus 2 \
                    --use_pretrained_model