from learn import learn
import astnn.learn as alg_astnn
import os
import time
import traceback
from expreport import report

datasets_path = "" # REPLACE <repo_path>

def try_learn(algo_name, algo, dataset, vis, datasets_path, epochs, lazy_mode, retrain):

    print("="*80)
    print("="*80)
    print("Running learning for:")
    print("     Algorithm: %s" % algo_name)
    print("     Dataset:   %s" % dataset)
    print("     Visual.:   %s" % vis)
    report.alg = algo_name
    report.dataset = dataset
    report.vis = vis
    print("="*20)
    print()
    try:
        dat_path = "%s/%s/fragments.csv" % (datasets_path, dataset)
        pairs_path = "%s/%s/pairs.csv" % (datasets_path, dataset)
        if lazy_mode:
            img_path = "%s/images/%s" % (datasets_path, vis)
        else:
            img_path = "%s/%s/images/%s" % (datasets_path, dataset, vis)

        algo(dat_path, img_path, pairs_path, epochs, retrain)
    except:
        print("="*40)
        print("Failed to learn using alg: %s %s %s retrain:%s" % (algo_name, dataset, vis, retrain))
        print(traceback.format_exc())

    print("="*80)
    print("="*80)
    print("="*80)
    print("="*80)
    print()
    print()

lazy_mode = True # since all current datasets are based on the same data, we can render images per id only once
epochs = 1
algos = {"ccd": learn,
         "astnn": alg_astnn.learn}

start_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())

# Note: the exp1-3 do NOT correspond exactly to the experiments, some data has to be split from the resulting csv files to reproduce the experiment results 
def exp1():
    report.set_save_file("experiment1_ccd_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        for alg in ["ccd", "astnn"]: # ccd includes all 3 binary classifiers, the results need to be filtered from the csv accordingly
            for vis in ["as"]:
                for dataset in ["astnn_t4"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode, False)
    report.reset()

exp1()

def exp2():
    report.set_save_file("experiment2_ccd_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        for alg in ["ccd"]:
            for vis in ["st", "sh", "kp", "as"]:
                for dataset in ["ds_with_duplicates", "ds_no_duplicates"]:
                    try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode, False)
    report.reset()

exp2()

def all(): # this runs all possible combinations, enable/disable the calls to the method according to your needs
    report.set_save_file("experiment_all_ccd_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        for alg in ["ccd", "astnn"]:
            for dataset in ["astnn", "ds_with_duplicates", "ds_no_duplicates"]:
                for vis in ["st", "sh", "kp", "as"]:
                    for retrain in [True, False]:
                        try_learn(alg, algos[alg], dataset, vis, datasets_path, epochs, lazy_mode, retrain)
    report.reset()

#all()

