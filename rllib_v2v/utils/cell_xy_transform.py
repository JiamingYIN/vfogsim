from pyproj import Proj

netOffset = [-385174.69, -6671039.23]

convBoundary = [0.00, 0.00, 1135.02, 1149.85]

origBoundary = [24.914235, 60.156132, 24.966609, 60.189319]

projParameter = "+proj=utm +zone=35 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

proj = Proj(projParameter)

x = [385340.63, 385361.66, 385898.90, 385499.54, 385479.09, 385669.00, 385799.40, 385365.18, 386236.36, 385783.29,
     386089.94, 386099.25]

y = [6671538.18, 6672004.81, 6671315.40, 6671845.21, 6671657.34, 6671283.40, 6672111.08, 6671253.94, 6671518.97,
     6671536.61, 6671777.39, 6671970.24]

for i in range(len(x)):
    print(f'*.gNodeB_[{i}].mobility.initialX = {x[i] + netOffset[0]}m')

    print(f'*.gNodeB_[{i}].mobility.initialY = {convBoundary[3] - y[i] - netOffset[1]}m')

    print(f'*.gNodeB_[{i}].mobility.initialZ = 50m')