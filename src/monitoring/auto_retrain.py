import pandas as pd
import sys
import subprocess

from drift_detection import DriftDetector


old_data = pd.read_csv(
    "data/processed/superstore_processed.csv"
)

# simulate new incoming data
new_data = old_data.copy()

# Simulate sales increase trend
new_data['Sales'] = (
    new_data['Sales'] * 1.8
)

# Simulate higher discounts
new_data['Discount'] = (
    new_data['Discount'] + 0.3
)

# Recalculate profit
new_data['Estimated_Profit'] = (
    new_data['Sales'] * 0.25
)

detector=DriftDetector()

drift = detector.detect(
    old_data,
    new_data
)

if drift:

    print(
        "\nRetraining Triggered..."
    )

    subprocess.run(
        [
            sys.executable,
            "src/models/train.py"
        ]
    )

else:

    print(
        "\nNo retraining needed"
    )