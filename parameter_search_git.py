import os
import json
import subprocess
from subprocess import call
from multiprocessing import Pool
import random

process = "python -W ignore train_mtl_git.py --exp_name mtl_sw_{} --use_gpu --gpu_id {} --encoder {} --save_dir {} --load_pickle {} {} --activation {} --l2 --wd {} --clip --save_policy loss --final_test"
def run(args) :
	subprocess.call(process.format(*args).split(), stdout=subprocess.PIPE)
	
i = 0
model_stats = []
ENCODER = ["bertweet", "roberta", "bert"]
LR = ["1e-5", "2e-5 --use_scheduler", "5e-5 --use_scheduler"]
BATCH_SIZE = [16, 32]
EMPATH = ['', '--use_empath']
WD = [0.1, 0.01, 0.001]

for encoder in ENCODER:
	save_dir = encoder + "_SenWave_mtl"
	dataset = "dataset_" + encoder + ".pkl"
	for empath in EMPATH :
		for wd in WD :
			print(f"Running instances {i+1} {i+2} out of {2*len(ENCODER)*len(EMPATH)*len(WD)}")  
			args_list = [(i+1, 1, encoder, save_dir, dataset, empath, "bce", wd),
						(i+2, 1, encoder, save_dir, dataset, empath, "tanh", wd)]
			p = Pool(processes=2)
			p.map(run, args_list)
			with open(f"{save_dir}/mtl_sw_{i+1}/test.json","r") as f :
				model_stats.append(json.load(f))
			with open(f"{save_dir}/mtl_sw_{i+2}/test.json","r") as f :
				model_stats.append(json.load(f))
			i+=2
