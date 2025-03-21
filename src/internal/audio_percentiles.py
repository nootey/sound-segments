import numpy as np


class AudioPercentiles:
    def __init__(self):
        print("AudioPercentiles initialized")

    def calculate_percentiles(self, entry):
        try:
            new_entry = {}

            # Process LAeq percentiles
            laeq_str = entry.get("LAeq [dBA] [s]")
            if laeq_str:
                method = "inverted_cdf"
                print(f"     Using method: {method}")

                laeq_values = np.array(list(map(float, laeq_str.split(","))))
                percentiles = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
                la_percentiles = {
                    f"LA{p} [dBA]": round(np.percentile(laeq_values, 100 - p, method=method), 2)
                    for p in percentiles
                }
                new_entry.update(la_percentiles)

            # Add LAF max and min
            laf_str = entry.get("LAF [dBA] [s]")
            if laf_str:
                laf_values = np.array(list(map(float, laf_str.split(","))))
                new_entry["LAF max [dBA"] = round(np.max(laf_values), 2)
                new_entry["LAF min [dBA"] = round(np.min(laf_values), 2)

            return new_entry
        except Exception as e:
            raise e