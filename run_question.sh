rm -rf question_checkpoints_1/
nohup torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29501 --nproc_per_node 2 noisy_learning_question.py --output_dir question_checkpoints_1/ > agree_only_all_1.txt & 
rm -rf question_checkpoints_2/
nohup torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29502 --nproc_per_node 2 noisy_learning_question.py --output_dir question_checkpoints_2/ > agree_only_all_2.txt & 
# rm -rf question_checkpoints_3/
# nohup torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29503 --nproc_per_node 2 noisy_learning_question.py --output_dir question_checkpoints_3/ > pm_all_3.txt & 