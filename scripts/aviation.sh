# single version 
python -u main_informer.py --model informer --data Aviation --root_path "./data/_data/aviation/01/" --features M --freq m --batch_size 4 --d_model 64 --d_ff 128 --train_epochs 200 --patience 70 --learning_rate 0.0001 --seq_len 24 --label_len 12 --pred_len 6 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --do_predict


# hpo version: batch_size, d_model, d_ff, seq_len, label_len are auto-tuned and independent of user's inputs.
python -u informer_aviation_hpo.py --model informer --data Aviation --root_path "./data/_data/aviation/01/" --checkpoints './checkpoints_hpo/01/'  --features M --freq m --batch_size 4 --d_model 64 --d_ff 128 --train_epochs 200 --patience 70 --learning_rate 0.0001 --seq_len 24 --label_len 12 --pred_len 6 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 1 --do_predict