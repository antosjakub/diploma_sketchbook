import os

items = os.listdir()

import os
import json

def get_data(item_path, item, file_name):
    json_path = os.path.join(item_path, file_name)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
            print(f"Loaded report from: {item}")
        except Exception as e:
            print(f"Error loading {item}/report.json: {e}")
    else:
        print(f"!! No report.json found in: {item}")

current_dir = 'gridsearch__2026-07-07--12-36-01'
data_reports = {}
data_config = {}
report_names = []
data_full = {}
# what to extract
term = 'pde'
# Get all items in current directory
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    if os.path.isdir(item_path):
        # config
        data_full[item] = {}
        print(item_path)
        data = get_data(item_path, item, 'model_metadata.json')['args']
        if data['prevent_resampling'] == True:
            data_full[item]["lambda_ic"] = data["lambda_ic"]
            data_full[item]["bs"] = data["bs"]
            data = get_data(item_path, item, 'report.json')
            data_full[item]["linf"] = data["test_linf"][term]
            data_full[item]["rel_l2"] = data["test_rel_l2"][term]
            data_full[item]["runtime"] = data["runtime"]
            report_names.append(item)

print(f"\nTotal reports loaded: {len(report_names)}")


print(list(data_full.values()))

# Create DataFrame
def create_pivot_table(err_type):
    import pandas as pd
    df = pd.DataFrame(list(data_full.values()))
    pivot = df.pivot_table(
        index='lambda_ic',
        columns='bs',
        values=err_type,
        aggfunc='first' # in case of duplicates
    )
    pivot = pivot.sort_index().sort_index(axis=1)

    #print("=== L2 Grid Search Results ===")
    #print(pivot.round(6))
    return pivot


import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(pivot, err_type, title, file_name, cmap='YlOrRd'):
    data = pivot.to_numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap=cmap)
    
    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    ax.set_title('Grid Search')
    ax.set_xlabel('bs')
    ax.set_ylabel('lambda_ic')
    
    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i,j]):
                ax.text(j, i, f'{data[i,j]:.4f}', ha='center', va='center', color='black')
    
    plt.colorbar(im, label=err_type)
    plt.tight_layout()
    plt.savefig(file_name)
    #plt.show()
    plt.close()


err_type = 'linf'
pivot = create_pivot_table(err_type)
plot_matrix(pivot, err_type, err_type, f'analysis_{err_type}')
err_type = 'rel_l2'
pivot = create_pivot_table(err_type)
plot_matrix(pivot, err_type, err_type, f'analysis_{err_type}')


