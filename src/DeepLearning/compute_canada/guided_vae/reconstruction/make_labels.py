import pandas as pd, torch, os

### MAKE LABELS FOR TESTING ###

# ---- paths you described ----
CSV_PATH  = "/home/athena/3DVAE-AgeDisentangled/preprocessing_data/friday_all_datasets.csv"
MESH_DIR  = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17"
OUT_PATH  = os.path.join(MESH_DIR, "raw/labels.pt")

print("CSV:", CSV_PATH)
print("Mesh dir:", MESH_DIR)
print("Output labels_test.pt:", OUT_PATH)

# ---- load CSV ----
df = pd.read_csv(CSV_PATH)

# helper: map id -> key (basename without extension)
def key_from_id(id_val):
    if pd.isna(id_val):
        return None
    return os.path.splitext(os.path.basename(str(id_val)))[0]

if "id" not in df.columns:
    raise RuntimeError(f"'id' column not found in CSV columns: {df.columns.tolist()}")

df["key"] = df["id"].apply(key_from_id)
df_lookup = df.set_index("key")

labels = {}
missing = []

# ---- iterate over meshes we actually have ----
for fname in os.listdir(MESH_DIR):
    # keep only mesh files; adjust extensions if needed
    if not fname.lower().endswith((".ply")):
        continue

    key = os.path.splitext(fname)[0]

    if key not in df_lookup.index:
        missing.append(key)
        continue

    row = df_lookup.loc[key]

    ### AGE ###
    if "age" not in row.index:
        raise RuntimeError(f"'age' column not found in CSV; available: {row.index.tolist()}")
    age_raw = row["age"]

    # skip if NaN / empty string
    if pd.isna(age_raw) or (isinstance(age_raw, str) and not age_raw.strip()):
        continue

    age = float(age_raw)  # allows decimals

    # ### FNAME ###
    # if "id" not in row.index:
    #     raise RuntimeError(f"'id' column not found in CSV; available: {row.index.tolist()}")
    # id_raw = row["id"]

    # # skip if NaN / empty string
    # if pd.isna(id_raw) or (isinstance(id_raw, str) and not id_raw.strip()):
    #     continue

    # id_raw = id_raw.replace("f_", "")

    # id = int(id_raw)  # allows decimals

    # ### GENDER ###
    # if "Gender" not in row.index:
    #     raise RuntimeError(f"'gender' column not found in CSV; available: {row.index.tolist()}")
    # gender_raw = row["Gender"]

    # # if NaN / empty string -> store empty string instead of skipping
    # if pd.isna(gender_raw) or (isinstance(gender_raw, str) and not gender_raw.strip()):
    #     gender = ""
    # else:
    #     gender = str(gender_raw)

    # ### DATASET ###
    # if "Dataset" not in row.index:
    #     raise RuntimeError(f"'dataset' column not found in CSV; available: {row.index.tolist()}")
    # dataset_raw = row["Dataset"]

    # # if NaN / empty string -> store empty string instead of skipping
    # if pd.isna(dataset_raw) or (isinstance(dataset_raw, str) and not dataset_raw.strip()):
    #     dataset = ""
    # else:
    #     dataset = str(dataset_raw)

    labels[key] = [0, 0.0, age]

# ---- save labels ----
torch.save(labels, OUT_PATH)
print(f"\nSaved {len(labels)} label entries to {OUT_PATH}")
if missing:
    print(f"{len(missing)} meshes had no CSV entry, first few missing keys: {missing[:10]}")


### READ LABLES BACK (for sanity check) ###

# path = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17/labels.pt"
# print("Loading:", path)
# labels = torch.load(path)

# print("Type:", type(labels))
# print("Number of entries:", len(labels))

# # show a few keys
# keys = list(labels.keys())
# print("First 5 keys:", keys[:5])

# # show the first few entries
# for k in keys[:5]:
#     print(f"{k!r} -> {labels[k]}")