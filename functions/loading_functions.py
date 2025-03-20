# Import libraries
import pandas as pd
from pathlib import Path

# INPUT_DIRS
INPUT_DATA_DIR = Path('.')

## Drop the Folder if it already exists
DATASETS_DIR = Path('dataset')

# Image & labels directory
TRAIN_IMAGES_DIR = DATASETS_DIR / 'images' / 'train'
TRAIN_LABELS_DIR = DATASETS_DIR / 'labels'/ 'train'
TEST_IMAGES_DIR = DATASETS_DIR / 'images' / 'test'

path_to_explore = DATASETS_DIR / 'images'
def merge_paths(base_path, relative_path):
    return base_path.replace('dataset/images', '') + relative_path

def get_dataset(type='train'):
    if type == 'train':
        df = pd.read_csv(INPUT_DATA_DIR / 'Train.csv')

        # The correct mapping from class to class_id
        class_map = {cls: i for i, cls in enumerate(sorted(df['class'].unique().tolist()))}

        # Map it
        df['class_id'] = df['class'].map(class_map)
    else:
        df = pd.read_csv(INPUT_DATA_DIR / 'Test.csv')
    df['ImagePath'] = df['ImagePath'].copy() # create a copy of the column.
    df['ImagePath'] = [merge_paths(str(path_to_explore), x) for x in df['ImagePath']]

    
    return df
# Load train and test files
train_dataset = pd.read_csv(INPUT_DATA_DIR / 'Train.csv')
test_dataset = pd.read_csv(INPUT_DATA_DIR / 'Test.csv')
submission_file = pd.read_csv(INPUT_DATA_DIR / 'SampleSubmission.csv')