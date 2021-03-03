### Setting up:

```bash
pip install -r requirements.txt
```

<!-- ### Preparing dataset:

**add the train, val and test csv files to data/ folder** -->

### How to run:

#### For scraping/hydrating:
```bash
python scrape.py -s True -q [queries] -l [limit on tweets]  
python scrape.py -H True -f [files containing tweets ids]

Note : The -H stands for hydration, and -s for scraping. Restrictions related to coordinates, time intervals, can be modified directly in the script.
```

<!-- #### For training :
```bash
python train.py --exp_name (value) --encoder (value) --data_dir (value) --save_dir (value) --lr (value) --batch_size (value) --save_policy (value) --activation (value) --optim (value) --wd (value) --epochs (value) --seed (value) --use_gpu(to use gpu) --use_empath(to use empath) --l2(to use l2 reg.) --use_scheduler(to use sched) --use_dropout(to use dropout)
``` -->

<!-- #### For generating predictions :
```bash
python generate_predictions.py --gpu_id (gpu_id) --model_name (BERT/ROBERTA) --model_path (path to saved model) --output_path (path to save dir) --data (path to dir containing hydrated csv) --use_empath (y/n) --activation (tanh/bce)
``` -->
#### For plotting graphs :
Run all the cells of plot_graphs.ipynb
TEST COMMIT
<!-- ### Saving scheme:
- final_test=false, stores best validation epoch
- final_test=true, stores test result after training on best validation hp.json -->
