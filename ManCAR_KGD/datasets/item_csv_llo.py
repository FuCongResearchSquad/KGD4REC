import pandas as pd
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="Software",
        help="Name of the dataset under ../raw_llo/",
    )
    return parser.parse_args()


def process_item_csv():
    args = parse_args()
    dataset_name = args.dataset_name
    RAW_DATA_PATH = f"../raw_llo/{dataset_name}/"
    PRO_DATA_PATH = f"../processed_llo_graph/{dataset_name}/"
    TARGET_PATH = f"../processed_llo_graph/{dataset_name}/"
    os.makedirs(TARGET_PATH, exist_ok=True)
    
    print("Extracting all parent_asin from the training set to assign IDs...")

    train_path = os.path.join(PRO_DATA_PATH, f"{dataset_name}.train.csv")
    if not os.path.exists(train_path):
        print(f"Error: training set file not found: {train_path}")
        return

    train_df = pd.read_csv(train_path)

    if 'Yelp' in RAW_DATA_PATH:
        asins = set(train_df['business_id'].unique())
        asins.update(train_df['history'].str.split().explode().unique())
    else:
        asins = set(train_df['parent_asin'].unique())
        history_asins = train_df['history'].str.split().explode().unique()
        asins.update(history_asins)

    asins = {a for a in asins if pd.notnull(a)}

    sorted_asins = sorted(list(asins))
    asin_to_id = {asin: i + 1 for i, asin in enumerate(sorted_asins)}

    print(f"ID mapping built. The training set has {len(asin_to_id)} unique items.")

    print("Generating item.csv field content...")
    item_rows = []
    meta_dict = {}
    for p_asin in sorted_asins:
        i_id = asin_to_id[p_asin]
        m_data = meta_dict.get(p_asin, {})

        title = m_data.get('title', '')
        description = m_data.get('description', '')

        cats = m_data.get('categories', [])
        if isinstance(cats, list):
            if len(cats) > 0 and isinstance(cats[0], list):
                cat_str = ", ".join([str(i) for i in cats[0]])
            else:
                cat_str = ", ".join([str(i) for i in cats])
        else:
            cat_str = str(cats)

        text_content = f"Title: {title}\nDescription: {description}\nCategories: {cat_str}"

        item_rows.append({
            'parent_asin': p_asin,
            'item_id': i_id,
            'second_cate_id': 0,
            'third_cate_id': 0,
            'store_id': m_data.get('store_id', 0),
            'text': text_content,
            'text_emb': ""
        })

    item_df = pd.DataFrame(item_rows)
    output_file = os.path.join(TARGET_PATH, f"{dataset_name}.item.csv")
    item_df.to_csv(output_file, index=False)

    print(f"\nProcessing completed successfully.")
    print(f"The generated item table has {len(item_df)} rows.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    process_item_csv()