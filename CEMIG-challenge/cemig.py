from geopy.distance import geodesic
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor

trainX = [
    [552, 238], # a (red)
    [220, 568], # b (red)
    [660, 613], # c (red)

    
    [548, 598], # Usiminas (green)
    [489, 595], # Usiminas (blue)
    [611,  18], # BHTec prox (blue)
    [146, 224], # BHTec descida (blue)
    
    [459, 396], # BHTec med (blue)
    [468, 484], # BHTec longe (blue)
    [247, 522], # Aglomerado1 (green)
]

trainY = [
    [-19.884362, -43.975112], # a (red)
    [-19.880996, -43.978154], # b (red)
    [-19.875391, -43.977202], # c (red)

    
    [-19.877094, -43.978037], # Usiminas (green)
    [-19.877851, -43.978243], # Usiminas (blue)    
    [-19.884660, -43.974987], # BHTec prox (blue)
    [-19.884490, -43.975472], # BHTec descida (blue)

    [-19.883896, -43.975428], # BHTec med (blue)
    [-19.882713, -43.975915], # BHTec longe (blue)    
    [-19.881952, -43.977276], # Aglomerado1 (green)
]

validX = [[412,598]]
validY = [[-19.877751,-43.978796]]
probeF = [[510,523]]

regr = RandomForestRegressor(max_depth=20, random_state=3, n_estimators=200)

for size in range(3,len(trainX)+1):
    regr.fit(trainX[:size], trainY[:size])

    predY = regr.predict(probeF)
    euc_error = distance.euclidean(validY, predY)
    met_error = geodesic(validY, predY).meters

    print(validY, predY, euc_error, met_error)