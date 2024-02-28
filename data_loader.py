import os


def load_data(folder_path="Bundle1", extension=".txt"):
    asset_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            file_path = os.path.join(folder_path, filename)
            measurements = []

            with open(file_path, 'r') as file:
                lines = file.readlines()
                asset_name = lines[0].strip()

                for line in lines[2:]:
                    time, value = map(lambda x: int(x) if '.' not in x else float(x), line.strip().split())
                    measurements.append((time, value))

            asset_data[asset_name] = measurements

    return asset_data
