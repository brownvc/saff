import json
import pandas as pd
with open("../data/VIS_results/VIS_result.json") as f:
    vis_result = json.load(f)

with open("../data/ours_1018/ours_result.json") as f:
    ours_result = json.load(f)

ours_result = pd.DataFrame.from_dict(ours_result, orient="index")
print(ours_result)
assert False

scenes = ["Balloon1-2", "Balloon2-2", "DynamicFace-2", "Jumping", "playground", "Skating-2", "Truck-2", "umbrella"]
headers=["Method", "Scene", "Split", "ARI", "fg-ARI"]
#print(pandas.DataFrame(data, headers))



data = []
# ours data
arisss = []
fg_arisss = []
result = ours_result
for scene in scenes:
    ariss = []
    fg_ariss = []
    for split in result[scene]:
        aris = result[scene][split]["ARI"]
        ariss.append(sum(aris)/float(len(aris)))
        fg_aris = result[scene][split]["fg-ARI"]
        fg_ariss.append(sum(fg_aris)/float(len(fg_aris)))
        data.append(["ours", scene, split, ariss[-1], fg_ariss[-1]])
        print(data)
    arisss.append(sum(ariss)/float(len(ariss)))
    fg_arisss.append(sum(fg_ariss)/float(len(fg_ariss)))
    #data.append(["ours", scene, "mean", arisss[-1], fg_arisss[-1]])
#data.append("ours", "mean, ")


print('                                         ')
print('-----------------------------------------')
print('                                         ')
print(pandas.DataFrame(data, headers))
