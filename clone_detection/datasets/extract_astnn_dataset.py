from paths import *
import pandas as pd
import pandas.io.sql as pdsql
import psycopg2
import os

tool_path = "<repo_path>" # REPLACE <repo_path>

def extract(T4_only = False):
    clone_pairs_train = pd.read_pickle(astnn_path + '/data/java/train/blocks.pkl')#.sample(frac=0.05, random_state=42)
    clone_pairs_train["train"] = 1
    clone_pairs_test = pd.read_pickle(astnn_path + '/data/java/test/blocks.pkl')#.sample(frac=0.05, random_state=42)
    clone_pairs_test["train"] = 0
    code_ids = set()
    code_ids.update(clone_pairs_train["id1"].tolist())
    code_ids.update(clone_pairs_train["id2"].tolist())
    code_ids.update(clone_pairs_test["id1"].tolist())
    code_ids.update(clone_pairs_test["id2"].tolist())
    print("Overall different code snippets: %s" % len(code_ids))

    clone_pairs = pd.concat([clone_pairs_train, clone_pairs_test])
    clone_pairs = clone_pairs[["id1", "id2", "label", "train"]]
    clone_pairs.columns = ["id1", "id2", "type", "train"]
    if T4_only:
        clone_pairs = clone_pairs[clone_pairs["type"].isin([0, 5])]

    clone_pairs["type"].replace({4:3}, inplace=True)
    clone_pairs["type"].replace({5:4}, inplace=True)
    print("Overall code pairs: %s" % len(clone_pairs.index))

    code = pd.read_csv(astnn_path + '/data/java/bcb_funcs_all.tsv', sep="\t", header=None)
    code.columns = ["id", "code"]
    print("DEBUG code lenA: %s" % len(code.index))
    code = code[code["id"].isin(code_ids)]
    print("DEBUG code lenB: %s" % len(code.index))

    id_func_sets = []
    #TODO change password of user sa to sa (empty by default but no empty password allowed by psycopg2)
    # run the database using: (replace <bcb_path>)
    #     java -cp h2-*.jar org.h2.tools.Server -baseDir <bcb_path>/bigclonebenchdb -ifExists
    conn = psycopg2.connect("dbname=bcb user='sa' password='sa' host='localhost' port=5435")
    query = "SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, FUNCTIONALITY_ID FROM CLONES"

    dat = pdsql.read_sql_query(query, conn)
    dat.head()
    ds1 = dat[["function_id_one", "functionality_id"]].copy().drop_duplicates(subset='function_id_one', keep='first')
    ds1.head()
    ds1.columns = ["id", "func"]
    ds1.head()
    id_func_sets.append(ds1)

    ds2 = dat[["function_id_two", "functionality_id"]].copy().drop_duplicates(subset='function_id_two', keep='first')
    ds2.columns = ["id", "func"]
    id_func_sets.append(ds2)


    query = "SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, FUNCTIONALITY_ID FROM FALSE_POSITIVES"
    dat = pdsql.read_sql_query(query, conn)
    ds3 = dat[["function_id_one", "functionality_id"]].copy().drop_duplicates(subset='function_id_one', keep='first')
    ds3.columns = ["id", "func"]
    id_func_sets.append(ds3)

    ds4 = dat[["function_id_two", "functionality_id"]].copy().drop_duplicates(subset='function_id_two', keep='first')
    ds4.columns = ["id", "func"]
    id_func_sets.append(ds4)

    id_funcs = pd.concat(id_func_sets).drop_duplicates(subset='id', keep='first')

    code = code.merge(id_funcs, on="id", how="left")

    def get_pair_func(row):
        val = 0
        try:
            val = (code[code["id"] == row["id1"]])["func"].iloc[0]
        except:
            print("WARN did not find func for id: ", row["id1"])
        return val
    clone_pairs["func"] = clone_pairs.apply(get_pair_func, axis=1)

    print("==== Clone pair function distribution")
    print(clone_pairs["func"].value_counts())
    print()

    print("==== Code snippet function distribution")
    print(code["func"].value_counts())

    return clone_pairs, code


if __name__ == "__main__":
    folder_path = tool_path + "/datasets/astnn_t4"
    os.makedirs(folder_path, exist_ok=True)
    pairs, funcs = extract()
    funcs.to_csv(folder_path + "/funcs.csv")
    pairs.to_csv(folder_path + "/pairs.csv")
