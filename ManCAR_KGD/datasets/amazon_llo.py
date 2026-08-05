import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="Software",
        help="Name of the dataset under ../raw_llo/",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name

    # Load the train/validation/test splits for the selected dataset.
    train_df = pd.read_csv(f'../raw_llo/{dataset_name}/{dataset_name}.train.csv')
    valid_df = pd.read_csv(f'../raw_llo/{dataset_name}/{dataset_name}.valid.csv')
    test_df = pd.read_csv(f'../raw_llo/{dataset_name}/{dataset_name}.test.csv')

    # Keep only training rows with a non-null interaction history.
    print(f'train_df before nall filter: {train_df.count()}')
    train_df = train_df[train_df['history'].notnull()]
    print(f'train_df after nall filter: {train_df.count()}')

    # Build the vocabulary of items observed in the training split, including
    # both target items and items appearing in interaction histories.
    train_asin = set(train_df['parent_asin'].unique())
    train_asin.update(train_df['history'].str.split().explode().unique())

    # Keep only validation rows with a non-null interaction history.
    print(f'valid_df before nall filter: {valid_df.count()}')
    valid_df = valid_df[valid_df['history'].notnull()]
    print(f'valid_df after nall filter: {valid_df.count()}')
    # Remove validation rows whose target item was never seen in training.
    valid_df = valid_df[valid_df['parent_asin'].apply(lambda x: x in train_asin)]
    print("finish filter parent_asin not in train_df")
    print(f'valid_df after parent_asin filter: {valid_df.count()}')
    # Remove validation rows whose history contains unseen items.
    valid_df = valid_df[valid_df['history'].apply(lambda x: all([his in train_asin for his in x.split()]))]
    print("finish filter history not in train_df")
    print(f'valid_df after history filter: {valid_df.count()}')

    # Keep only test rows with a non-null interaction history.
    print(f'test_df before nall filter: {test_df.count()}')
    test_df = test_df[test_df['history'].notnull()]
    print(f'test_df after nall filter: {test_df.count()}')
    # Remove test rows whose target item was never seen in training.
    test_df = test_df[test_df['parent_asin'].apply(lambda x: x in train_asin)]
    print("finish filter parent_asin not in train_df")
    print(f'test_df after parent_asin filter: {test_df.count()}')
    # Remove test rows whose history contains unseen items.
    test_df = test_df[test_df['history'].apply(lambda x: all([i in train_asin for i in x.split()]))]
    print("finish filter history not in train_df")
    print(f'test_df after history filter: {test_df.count()}')

    # Save the filtered splits to the processed dataset directory.
    os.makedirs(f'../processed_llo_graph/{dataset_name}', exist_ok=True)
    train_df.to_csv(f'../processed_llo_graph/{dataset_name}/{dataset_name}.train.csv', index=False)
    valid_df.to_csv(f'../processed_llo_graph/{dataset_name}/{dataset_name}.valid.csv', index=False)
    test_df.to_csv(f'../processed_llo_graph/{dataset_name}/{dataset_name}.test.csv', index=False)


if __name__ == "__main__":
    main()