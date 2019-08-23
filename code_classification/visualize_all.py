import sys
sys.path.append("../visualization_algorithms")
import pandas as pd
import simpletext.alg as alg_st
import keywords_picto.alg as alg_kp
import synth_high.alg as alg_sh
import ast_simple.alg as alg_as
import os
import traceback
import multiprocessing
import threading
from PIL import Image


ds_path = "<repo_path>/clone_classification/datasets" #REPLACE <repo_path>

NB_CORES = multiprocessing.cpu_count()
fail_lock = threading.Lock()

def visualize(algorithm, lang, code, out_path):
    try:
        img = algorithm(code, lang)
        img.save(out_path)

    except:
        fail_lock.acquire()
        try:
            print("generating image %s-%s: %s failed" % (dataset, algo, cid))
            print(traceback.format_exc())
            print("generating default image as replacement")

            ###########################
            background = (128, 128, 128)
            img = Image.new('RGBA', (448, 448), background)
            img.save(out_path)
            ###########################
        finally:
            fail_lock.release()


def visualize_ds(algo, lang, dataset, datasets_path, lazy_mode=False):
    code_ds = pd.read_csv(datasets_path + "/" + dataset + '/fragments.csv')
    if algo == "as":
        vis = alg_as.generate_viz
    elif algo == "sh":
        vis = alg_sh.render
    elif algo == "kp":
        vis = alg_kp.keywords_picto
    elif algo == "st":
        vis = alg_st.text2png
    else:
        print("Unknown algorithm:", algo)

    if lazy_mode:
        out_dir = "%s/images/%s" % (datasets_path, algo)
    else:
        out_dir = "%s/%s/images/%s" % (datasets_path, dataset, algo)

    os.makedirs(out_dir, exist_ok=True)
    work = []
    for _, r in code_ds.iterrows():
        cid = r["id"]
        out_path = "%s/%s.png" % (out_dir, cid)
        if lazy_mode:
            if os.path.exists(out_path):
                continue
            else:
                work.append((vis, lang, r["code"], out_path))

    with multiprocessing.Pool(processes=(NB_CORES-1)) as pool: # auto closing workers
        pool.starmap(visualize, work)



def try_visualize(algo, lang, dataset, datasets_path, lazy_mode):
    try:
        visualize_ds(algo, lang, dataset, datasets_path, lazy_mode)
    except:
        print("="*40)
        print("Failed to visualize dataset: %s using algorithm %s" % (dataset, algo))
        print(traceback.format_exc())

    print("="*80)

lazy_mode = True #since all current datasets are based on the same data, we can render images per id only once

try_visualize("st", "C", "oj", ds_path, lazy_mode)
