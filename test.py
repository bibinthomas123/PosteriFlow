import pickle
from pathlib import Path

data_dir = Path("data/training_data")  # replace with your actual data dir

with open(data_dir / 'training_scenarios.pkl', 'rb') as f:
    scenarios = pickle.load(f)

with open(data_dir / 'baseline_results.pkl', 'rb') as f:
    baseline_results = pickle.load(f)

# Print types and lengths
print("Scenarios type:", type(scenarios), "Length:", len(scenarios))
print("Baseline results type:", type(baseline_results), "Length:", len(baseline_results))

# If dict, print first 5 keys
if isinstance(scenarios, dict):
    print("Scenario keys sample:", list(scenarios.keys())[:5])
if isinstance(baseline_results, dict):
    print("Baseline keys sample:", list(baseline_results.keys())[:5])

# If list, print first 2 items to see structure
if isinstance(scenarios, list):
    print("Scenario item sample:", scenarios[:2])
if isinstance(baseline_results, list):
    print("Baseline item sample:", baseline_results[:2])
