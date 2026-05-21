import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:

    def detect(self, old_df, new_df):

        numerical = old_df.select_dtypes(
            include=["number"]
        ).columns

        drift_found=False

        for col in numerical:

            stat,p = ks_2samp(
                old_df[col],
                new_df[col]
            )

            print("\nFeature:",col)
            print("p-value:",p)

            if p < 0.05:

                print("Drift Detected")
                drift_found=True

            else:

                print("No Drift")

        return drift_found