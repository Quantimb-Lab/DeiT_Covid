#!your_python_pth

import os

os.system("python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py "
          "--model deit_base_distilled_patch16_224"
          " --distillation-type hard --batch-size 16 "
          "--data-path ./data/"
          " --data-set Covid --teacher-model densenet169 "
          "--teacher-path ./teacher_model/teacher")