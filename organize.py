
import os
import pandas as pd
from utils.labels import parse_crema_label, parse_savee_label, parse_tess_label

DATASET_PATHS = {
    'crema': 'audioFiles/crema',
    'savee': 'audioFiles/savee',
    'tess': 'audioFiles/tess',
}

def build_metadata():
    rows = []
    for dataset_name, path in DATASET_PATHS.items():
        for file in os.listdir(path):
            if not file.endswith('.wav'):
                continue
            full_path = os.path.join(path, file)

            if dataset_name == 'crema':
                emotion = parse_crema_label(file)
            elif dataset_name == 'savee':
                emotion = parse_savee_label(file)
            elif dataset_name == 'tess':
                emotion = parse_tess_label(file)
            else:
                continue

            rows.append({
                'path': full_path,
                'emotion': emotion,
                'dataset': dataset_name
            })

    df = pd.DataFrame(rows)
    df.to_csv('dataset/metadata.csv', index=False)
    print(" metadata.csv created with", len(df), "entries.")

if __name__ == '__main__':
    build_metadata()