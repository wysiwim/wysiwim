from learn import learn
import os
import time
import traceback
from expreport import report

DATASETS_PATH = "<repo_path>/clone_classification/datasets"  # REPLACE <repo_path>

def try_learn(algo_name, algo, dataset, vis, datasets_path, model_selection, epochs, lazy_mode):

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
        dat_path = "%s/%s/fragments.csv" % (DATASETS_PATH, dataset)
        if lazy_mode:
            img_path = "%s/images/%s" % (DATASETS_PATH, vis)
        else:
            img_path = "%s/%s/images/%s" % (datasets_path, dataset, vis)

        algo(dat_path, img_path, model_selection, epochs)
    except:
        print("="*40)
        print("Failed to learn using alg: %s %s %s" % (algo_name, dataset, vis))
        print(traceback.format_exc())

    print("="*80)
    print("="*80)
    print("="*80)
    print("="*80)
    print()
    print()

lazy_mode = True
epochs = 21

start_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.gmtime())

def exp1():
    report.set_save_file("experiment1_cc_%s.csv" % start_time_str)
    for run in range(3):
        report.run = run
        print("-CSV-Run: %s" % run)
        for resnet in ["resnet18", "resnet50"]:
            for vis in ["st"]:
                for dataset in ["oj"]:
                    try_learn("cc", learn, dataset, vis, datasets_path, resnet, epochs, lazy_mode)
    report.reset()

exp1()

