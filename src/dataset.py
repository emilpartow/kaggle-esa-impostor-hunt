# src/dataset.py

import os
import pandas as pd
import re


def get_project_dirs():
    """
    Automatically determine the project root and the data directory.

    Returns
    -------
    tuple[str, str]
        (root_dir, data_dir) where:
        - root_dir is the project root (one level above this file),
        - data_dir is <root_dir>/data

    Notes
    -----
    - If __file__ is not defined (e.g., interactive execution), fall back to "..".
    """
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        # Fallback: executed in an environment where __file__ is not set
        root_dir = os.path.abspath("..")

    data_dir = os.path.join(root_dir, "data")
    return root_dir, data_dir


def build_pairwise_dataframe(data_dir, split="train"):
    """
    Load all text pairs for the given split ('train' or 'test') and return them as a DataFrame.
    For 'train', also include the 'real_text_id' label if present in the CSV.

    Parameters
    ----------
    data_dir : str
        Path to the data directory (must contain <split>.csv and the <split>/ subfolder).
    split : {'train', 'test'}, optional
        Dataset split to load. Defaults to 'train'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - id:            article identifier
        - text_A:        content of file_1.txt
        - text_B:        content of file_2.txt
        - real_text_id:  (only for split='train' and if present in the CSV)
    """
    # Load the CSV for the split (e.g., data/train.csv or data/test.csv)
    csv_path = os.path.join(data_dir, f"{split}.csv")
    df = pd.read_csv(csv_path)

    # Folder containing the per-article subfolders (e.g., data/train/article_0001/)
    articles_folder = os.path.join(data_dir, split)

    text_a_list = []
    text_b_list = []

    # Iterate row-by-row to collect the paired texts
    for _, row in df.iterrows():
        art_id = row["id"]
        article_path = os.path.join(articles_folder, f"article_{art_id:04d}")
        file_1_path = os.path.join(article_path, "file_1.txt")
        file_2_path = os.path.join(article_path, "file_2.txt")

        # Read file contents (UTF-8). Ignore decoding errors to avoid crashes on rare encoding issues.
        with open(file_1_path, encoding="utf-8", errors="ignore") as f:
            text_a = f.read()
        with open(file_2_path, encoding="utf-8", errors="ignore") as f:
            text_b = f.read()

        text_a_list.append(text_a)
        text_b_list.append(text_b)

    # Assemble the final DataFrame
    new_df = pd.DataFrame({
        "id": df["id"],
        "text_A": text_a_list,
        "text_B": text_b_list
    })

    # For training, propagate the label if it is present in the CSV
    if split == "train" and "real_text_id" in df.columns:
        new_df["real_text_id"] = df["real_text_id"]

    return new_df


def build_test_csv(data_dir):
    """
    Create data/test.csv from the folder names in data/test/.

    This scans data/test/ for subfolders named 'article_<id>' and writes a CSV
    with a single 'id' column sorted ascending.

    Parameters
    ----------
    data_dir : str
        Path to the data directory (must contain the 'test' subfolder).

    Side Effects
    ------------
    Writes '<data_dir>/test.csv' and prints a short status message.
    """
    # Find article directories under data/test/
    test_path = os.path.join(data_dir, "test")
    article_dirs = [d for d in os.listdir(test_path) if d.startswith("article_")]

    # Extract numeric IDs from folder names like 'article_0001'
    article_ids = []
    for article in article_dirs:
        m = re.match(r"article_(\d+)", article)
        if m:
            article_ids.append(int(m.group(1)))

    article_ids = sorted(article_ids)

    # Build and save the CSV
    df = pd.DataFrame({"id": article_ids})
    out_path = os.path.join(data_dir, "test.csv")
    df.to_csv(out_path, index=False)

    print(f"Built test.csv with {len(df)} entries: {out_path}")


# --- Usage ---
if __name__ == "__main__":
    root_dir, data_dir = get_project_dirs()

    # Rebuild test.csv based on the directory structure (data/test/article_XXXX)
    build_test_csv(data_dir)

    # Load pairwise text data for both splits
    train_df = build_pairwise_dataframe(data_dir, split="train")
    test_df = build_pairwise_dataframe(data_dir, split="test")

    # Quick preview
    print(train_df.head())
    print(test_df.head())

    # Save back to CSV for quick later access (overwrites existing files)
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
