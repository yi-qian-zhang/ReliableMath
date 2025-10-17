python code/main_domain_shift.py --dataset polaris \
  --dataset polaris_20 \
  --prompt_dir /data/home/zyq/ReliableMath/prompt/domain_shift 

python code/main_pronoun_ambiguity.py \
  --dataset polaris_20 \
  --prompt_dir /data/home/zyq/ReliableMath/prompt/pronoun_ambiguity \
  --force

python code/main_remove.py --dataset polaris 