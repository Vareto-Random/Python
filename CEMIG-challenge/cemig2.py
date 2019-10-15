from geopy.distance import geodesic
from scipy.spatial import distance
from sklearn.ensemble import RandomForestRegressor

import cv2 as cv
import numpy as np

pixelsPhoto = np.array([
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
])

coords = np.array([
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
])

for size in range(3,len(pixelsPhoto)+1):
    homo, _ = cv.findHomography(pixelsPhoto[:size], coords[:size])

    Ephoto = [[412, 598, 1]]
    Ecoord = [[-19.877751, -43.978796, 1]]
    probeF = [[510, 523, 1]]

    result = np.dot(homo, np.transpose(probeF))
    normed =  (1/result[2]) * result

    euc_ds = np.sqrt((normed[0] - Ecoord[0][0])**2 + (normed[1] - Ecoord[0][1])**2)
    met_error = geodesic(Ecoord, normed).meters
    print(Ecoord, np.transpose(normed[0:2]), euc_ds[0], met_error)