from helpers.visualizer import plot_pareto_front
from warren_buffett import WarrenBuffett
import numpy as np


def reorder_weights(weights, asset_names, asset_index_map):
    reordered_weights = [0] * len(asset_order)
    for name, weight in zip(asset_names, weights):
        index = asset_index_map[name]
        reordered_weights[index] = weight
    return reordered_weights


def non_zero_indices(weights_list):
    # for i, weights in enumerate(weights_list):
    rounded_weights = np.round(weights_list, decimals=6)
    non_zero_indices = np.where(rounded_weights > 0)[0]
    return non_zero_indices
    # print(f"Non-zero indices for weights list: {non_zero_indices}")


def find_max_weight_index(weights_list):
    max_count = 0
    max_index = None
    for i, weights in enumerate(weights_list):
        count = np.sum(np.round(weights, decimals=3) > 0)
        if count > max_count:
            max_count = count
            max_index = i
    return max_index


def save_pareto_front(pareto_solutions, filename):
    with open(filename, 'w') as file:
        for _, _profit, _risk in pareto_solutions:
            formatted_profit = "{:,.6f}".format(_profit)
            formatted_risk = "{:,.6f}".format(_risk)
            file.write(f"{formatted_profit} {formatted_risk}\n")


wb_ecm = WarrenBuffett("ecm", "Bundle1")
wb_wsm = WarrenBuffett("wsm", "Bundle1")

pareto_solutions_ecm = wb_ecm.make_me_rich()
pareto_solutions_wsm = wb_wsm.make_me_rich()

save_pareto_front(pareto_solutions_ecm, "ecm_pareto.txt")
save_pareto_front(pareto_solutions_wsm, "wsm_pareto.txt")

# For pareto_solutions_ecm
max_weight_index_ecm = find_max_weight_index([solution[0] for solution in pareto_solutions_ecm])
print("Index of max weight list in pareto_solutions_ecm:", max_weight_index_ecm)

# For pareto_solutions_wsm
max_weight_index_wsm = find_max_weight_index([solution[0] for solution in pareto_solutions_wsm])
print("Index of max weight list in pareto_solutions_wsm:", max_weight_index_wsm)

# plot_pareto_front(pareto_solutions_wsm)
# plot_pareto_front(pareto_solutions_ecm)
# print(pareto_solutions_wsm[max_weight_index_wsm])
# print(pareto_solutions_ecm[max_weight_index_ecm])

# wb_ecm.plot_all_predictions()
# profile, profit, risk = wb_ecm.get_investment_profiles()[19]
# # indices = np.where(profile > 1e-7)[0]
# indices = np.argsort(np.abs(profile))[::-1][:5]
# print("Indices of the 5 most significant values: ", indices)
# rounded_profile = np.round(profile, decimals=2)
#
# print("Rounded Profile: ", rounded_profile)
# print("Profile: ", profile)
#
assets_names = np.array(wb_ecm.get_assets_keys())
print(pareto_solutions_ecm[50])
selected = non_zero_indices(pareto_solutions_ecm[50][0])
# print_non_zero_indices(pareto_solutions_ecm[50][0])
print(assets_names[selected])
print(np.sum(pareto_solutions_ecm[50][0]))
# selected_assets = assets_names[indices]
# wb_ecm.show_predictions_for_asset(selected_assets)

asset_order = [
    "SuperFuture", "Apples", "WorldNow", "Electronics123", "Photons",
    "SpaceNow", "PearPear", "PositiveCorrelation", "BetterTechnology",
    "ABCDE", "EnviroLike", "Moneymakers", "Fuel4", "MarsProject",
    "CPU-XYZ", "RoboticsX", "Lasers", "WaterForce", "SafeAndCare", "BetterTomorrow"
]

# Create a dictionary mapping asset names to their indices in the desired order
asset_index_map = {asset: index for index, asset in enumerate(asset_order, start=0)}


# Extract profit, risk, and weights from the selected portfolio
weights, profit, risk = pareto_solutions_ecm[50]

# Reorder the weights based on the desired order of assets
reordered_weights = reorder_weights(weights, assets_names, asset_index_map)

print(np.sum(np.round(reordered_weights, decimals=6)))

# Save the portfolio to a file
with open("selected_portfolio.txt", "w") as f:
    f.write(f"{profit:.6f}, {risk:.6f}, {' '.join([f'{w:.6f}' for w in reordered_weights])}")

print(assets_names)
print(pareto_solutions_ecm[50][0])