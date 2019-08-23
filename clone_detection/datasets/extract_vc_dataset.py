import psycopg2
import pandas as pd
import pandas.io.sql as pdsql
import os
import traceback
from sklearn.model_selection import train_test_split


tool_path = "<repo_path>"
# path to the "bcb_reduced" folder of the BigBloneBench
bcb_path = "<bceval_path>/ijadataset/bcb_reduced"


# ids of code that we dont want to use (e.g. due to non-compilability)
blacklisted_func_ids = [22648747]

def filter_blacklisted_pairs(ds):
    ds = ds[(~ds["id1"].isin(blacklisted_func_ids)) &
            (~ds["id2"].isin(blacklisted_func_ids))]
    return ds


def limited(ds, type, limits):
    if limits is None:
        return ds
    type_limit = limits[str(type)]
    if type_limit is None:
        return ds
    if type_limit >= len(ds.index):
        return ds
    else:
        return ds.sample(n=type_limit, random_state=42).copy()


def rename_cols(ds):
    ds.columns = ["id1", "id2", "func"]
    return ds


def generate_cross(s1, s2, limit, func_count):
    res = []
    count = 0
    for _, r1 in s1.iterrows():
        for _, r2 in s2.iterrows():
            count += 1
            if limit and (count > 2 * (limit / (func_count-1.0))):
                return res
            # attribute half of the non-clones to functionality of set1 and half to the other to simplify stratified splitting later
            func = r1["func"] if ((count % 2) == 0) else r2["func"]
            res.append({"id1": r1["id"], "id2": r2["id"], "func": func, "type": 0})
    return res


def assign_test_train(ds):
    train, test = train_test_split(ds, test_size=0.2, stratify=ds[["func"]], random_state=42)
    train = train.copy()
    test = test.copy()
    train["train"] = 1
    test["train"] = 0
    ds = pd.concat([train, test])
    ds.set_index(ds.columns[0])
    return ds.sort_index()


def generate_non_clones(code_set, limit=None):
    functionalities = code_set["func"].unique()
    #Note: .sample(frac=1) shuffles the datasets
    func_sets = [code_set[code_set["func"] == x].sample(frac=1, random_state=42) for x in functionalities]
    non_clones_list = []
    for i in range(len(functionalities)):
        for j in range(i+1, len(functionalities)):
            f1 = functionalities[i]
            f2 = functionalities[j]
            #print(f1, f2)
            ncs = generate_cross(func_sets[i], func_sets[j], limit, len(functionalities))
            non_clones_list.extend(ncs)

    return pd.DataFrame(non_clones_list)


def extract_id_to_func(ds):
    ds1 = ds[["id1", "func"]].copy().drop_duplicates(subset='id1', keep='first')
    ds1.columns = ["id", "func"]
    ds2 = ds[["id2", "func"]].copy().drop_duplicates(subset='id2', keep='first')
    ds2.columns = ["id", "func"]
    code_ids = pd.concat([ds1, ds2]).drop_duplicates(subset='id', keep='first').copy()
    return code_ids


def extract_clone_pairs(functionalities, remove_duplicates=True, limits=None, artifical_non_clones=True):
    #TODO change password of user sa to sa (empty by default but no empty password allowed by psycopg2)
    conn = psycopg2.connect("dbname=bcb user='sa' password='sa' host='localhost' port=5435")

    pairs = None

    for f in functionalities:
        clones = []
        query_start = 'SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, FUNCTIONALITY_ID FROM CLONES WHERE '

        if remove_duplicates:
            query = query_start + 'SYNTACTIC_TYPE=1 AND FUNCTIONALITY_ID=%s;' % f
            dat = rename_cols(pdsql.read_sql_query(query, conn))
            t1_code_ids = extract_id_to_func(dat)
            blacklisted_func_ids.extend(set(t1_code_ids["id"].tolist()))


        query = query_start + 'SYNTACTIC_TYPE=3 AND (SIMILARITY_LINE<=0.5) AND FUNCTIONALITY_ID=%s;' % f
        dat = rename_cols(pdsql.read_sql_query(query, conn))
        dat["type"] = 4
        dat = filter_blacklisted_pairs(dat)
        dat = limited(dat, 4, limits)
        clones.append(dat)

        if pairs is None:
            pairs = pd.concat(clones)
        else:
            pairs = pairs.append(pd.concat(clones))

    if not artifical_non_clones:
        for f in functionalities:
            query_start = 'SELECT FUNCTION_ID_ONE, FUNCTION_ID_TWO, FUNCTIONALITY_ID FROM FALSE_POSITIVES WHERE '

            query = query_start + 'FUNCTIONALITY_ID=%s;' % f
            dat = rename_cols(pdsql.read_sql_query(query, conn))
            dat["type"] = 0
            dat = filter_blacklisted_pairs(dat)
            dat = limited(dat, 0, limits)
            pairs = pairs.append(dat)

    code_ids = extract_id_to_func(pairs)

    def extract_code(code_id, functionality):
        query = 'SELECT TYPE, NAME, STARTLINE, ENDLINE FROM FUNCTIONS WHERE ID=%s;' % code_id
        dat = pdsql.read_sql_query(query, conn).iloc[0]
        with open(bcb_path + "/" + str(functionality) + "/" + dat["type"] + "/" + dat["name"], "r", encoding="utf8",
                  errors="ignore") as fin:
            i = 1
            code = []
            for l in fin:
                if i >= dat["startline"]:
                    code.append(l)
                if i >= dat["endline"]:
                    break
                i += 1
        return "".join(code)

    def get_code_by_row(row):
        return extract_code(row["id"], row["func"])

    code_ids["code"] = None
    code_ids["code"] = code_ids.apply(get_code_by_row, axis=1)

    if artifical_non_clones:
        nc_limit = None if (limits is None) or ("0" not in limits) else limits["0"]
        non_clones = generate_non_clones(code_ids, nc_limit)
        #print(len(non_clones.index))
        pairs = pairs.append(non_clones)

    for f_idx, f in enumerate(functionalities):
        print("====================")
        print("functionality: %s " % f)
        for ct in range(5):
            print("type-%s: %s pairs " % (ct, len(pairs[(pairs["func"] == f) & (pairs["type"] == ct)])))

    print("====================")
    for f_idx, f in enumerate(functionalities):
        print("functionality - %s : %s snippets" % (f, len(code_ids[code_ids["func"] == f])))

    return assign_test_train(pairs), code_ids


def extract_generalization_set(funcs_train, funcs_test, remove_duplicates, limits, artifical_non_clones):
    train_pairs, train_code = extract_clone_pairs(funcs_train, remove_duplicates, limits, artifical_non_clones)
    test_pairs, test_code = extract_clone_pairs(funcs_test, remove_duplicates, limits, artifical_non_clones)

    train_pairs["train"] = "1"
    test_pairs["train"] = "0"

    return pd.concat([train_pairs, test_pairs]), pd.concat([train_code, test_code])


def try_extract_vc(dataset_name, remove_duplicates=False, limits=None, artifical_non_clones=True):
    try:
        functionalities = [7, 13, 44]
        pairs, code = extract_clone_pairs(functionalities, remove_duplicates=remove_duplicates,
                                 limits=limits, artifical_non_clones=artifical_non_clones)

        folder_path = tool_path + "/datasets/%s" % dataset_name
        os.makedirs(folder_path, exist_ok=True)
        pairs.to_csv(folder_path + "/pairs.csv")
        code.to_csv(folder_path + "/fragments.csv")
    except:
        print("Failed to extract vc dataset %s" % dataset_name)
        print(traceback.format_exc())

    print()
    print("=" * 80)


if __name__ == "__main__":
	try_extract_vc("ds_with_duplicates",
		       remove_duplicates=False, limits={"0":500, "1":0, "2":0, "3":0, "4":500},
		       artifical_non_clones=False)
	try_extract_vc("ds_no_duplicates",
		       remove_duplicates=True, limits={"0":500, "1":0, "2":0, "3":0, "4":500},
		       artifical_non_clones=False)
