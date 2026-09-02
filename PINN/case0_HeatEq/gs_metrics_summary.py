import os
import json
import sys

current_dir = sys.argv[1]
#current_dir = 'gridsearch__2026-07-08--18-07-48'

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

report_names = []
data_full = {}
# what to extract
# Get all items in current directory
for item in os.listdir(current_dir):
    item_path = os.path.join(current_dir, item)
    if os.path.isdir(item_path):
        # config
        data_full[item] = {}
        #print(item)
        #data = get_data(item_path, item, 'model_metadata.json')['args']
        # report
        data = get_data(item_path, item, 'report.json')
        data_full[item]["runtime"] = data["runtime"]
        for term in ('pde', 'ic'):
            data_full[item][f"linf[{term}]"] = data["test_linf"][term]
            data_full[item][f"rel_l2[{term}]"] = data["test_rel_l2"][term]
        report_names.append(item)

print(f"\nTotal reports loaded: {len(report_names)}")

for name, data in data_full.items():
    print("======", name, "=======")
    print("  |  ".join([f'{k}={v:.4f}' for k,v in data.items()]))


