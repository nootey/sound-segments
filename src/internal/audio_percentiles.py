import numpy as np


class AudioPercentiles:
    def __init__(self):
        print("AudioPercentiles initialized")

    def calculate_percentiles(self, entry, keys=["LAeq [dBA] [s]"], label_prefix="LA"):
        try:
            new_entry = {}
            percentiles = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
            method = "inverted_cdf"
            new_entry["Percentile Method"] = method
            print(f"     Using method: {method}")

            # Process LAeq percentiles
            for key in keys:
                if key in entry:
                    print(f"     Processing: {key}")
                    values = np.array(list(map(float, entry[key].split(","))))
                    prefix = label_prefix if label_prefix else key.split()[0]
                    new_entry.update({
                        f"{prefix}{p} [dBA]": round(np.percentile(values, 100 - p, method=method), 2)
                        for p in percentiles
                    })

            # Add LAF max and min
            laf_str = entry.get("LAF [dBA] [s]")
            if laf_str:
                laf_values = np.array(list(map(float, laf_str.split(","))))
                new_entry["LAF max [dBA]"] = round(np.max(laf_values), 2)
                new_entry["LAF min [dBA]"] = round(np.min(laf_values), 2)

            return new_entry
        except Exception as e:
            raise e