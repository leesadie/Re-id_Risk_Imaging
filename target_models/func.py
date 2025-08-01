import json
from sklearn.model_selection import GroupShuffleSplit

def make_train_val_split(
    df, image_root, save_prefix="cnn", label_col="label", group_col="PatientID"
):
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42, n_splits=1)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    all_data = []
    for split_name, df_split in [("train", train_df), ("val", val_df)]:
        for _, row in df_split.iterrows():
            all_data.append({
                "Path":  row["Path"],
                "Age":   float(row["Age"]),
                "Sex":   row["Sex"],
                "Label": int(row[label_col]),
                "Split": split_name
            })

    with open(f"images/{save_prefix}_images.json", "w") as f:
        json.dump(all_data, f, indent=2)

    return train_df, val_df