
python preprocess_data.py --dataset dataset/cometa/ \
                          --train_data train.json \
                          --max_ent_len 64

python generate_candidates.py --dataset dataset/cometa/ \
                          --model model_retriever/cometa_retriever.pt \
                          --gpus 2
python evaluate.py --dataset dataset/cometa/ \
                    --model model_disambiguation/cometa_disambiguation_prompt_pretrain.pt \
                    --pretrained_model_path model_pretrain/cometa_pretrain.pt \
                    --gpus 2 \
                    --use_pretrained_model