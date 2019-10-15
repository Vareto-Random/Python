trainX = [
    [552, 238], # a (red)
    [220, 568], # b (red)
    [660, 613], # c (red)
    # [412, 598], # e (red)
    [247, 522], # Aglomerado1 (green)
    [335, 532], # Aglomerado2 (green)
    [710, 604], # Prédio torre gemea (green)
    [548, 604], # Usiminas (green)
    [489, 598], # Usiminas (blue)
    [411, 596], # Usiminas (red)
    [127, 560], # Abbot (blue)
    [611,  18], # BHTec prox (blue)
    [459, 396], # BHTec med (blue)
    [468, 484], # BHTec longe (blue)
    [146, 224], # BHTec descida (blue)
]

trainY = [
    [-19.884362, -43.975112], # a (red)
    [-19.880996, -43.978154], # b (red)
    [-19.875391, -43.977202], # c (red)
    # [-19.877751, -43.978796], # e (red)
    [-19.881952, -43.977276], # Aglomerado1 (green)
    [-19.881580, -43.977137], # Aglomerado2 (green)
    [-19.872947, -43.977497], # Prédio torre gemea (green)
    [-19.877094, -43.978037], # Usiminas (green)
    [-19.877851, -43.978243], # Usiminas (blue)
    [-19.877753, -43.978794], # Usiminas (red)
    [-19.881528, -43.978250], # Abbot (blue)
    [-19.884660, -43.974987], # BHTec prox (blue)
    [-19.883896, -43.975428], # BHTec med (blue)
    [-19.882713, -43.975915], # BHTec longe (blue)
    [-19.884490, -43.975472], # BHTec descida (blue)
]





coords = np.array([
    [-19.884362, -43.975112], # a (red)
    [-19.880996, -43.978154], # b (red)
    [-19.875391, -43.977202], # c (red)
    [-19.881952, -43.977276], # Aglomerado1 (green)
    # [-19.881580, -43.977137], # Aglomerado2 (green)
    # [-19.872947, -43.977497], # Prédio torre gemea (green)
    [-19.877094, -43.978037], # Usiminas (green)
    [-19.877851, -43.978243], # Usiminas (blue)
    # [-19.877753, -43.978794], # Usiminas (red)
    # [-19.881528, -43.978250], # Abbot (blue)
    [-19.884660, -43.974987], # BHTec prox (blue)
    [-19.883896, -43.975428], # BHTec med (blue)
    [-19.882713, -43.975915], # BHTec longe (blue)
    [-19.884490, -43.975472], # BHTec descida (blue)
])

pixelsPhoto = np.array([
    [552, 238], # a (red)
    [220, 568], # b (red)
    [660, 613], # c (red)
    [247, 522], # Aglomerado1 (green)
    # [335, 532], # Aglomerado2 (green)
    # [710, 604], # Prédio torre gemea (green)
    [548, 598], # Usiminas (green)
    [489, 595], # Usiminas (blue)
    # [411, 596], # Usiminas (red)
    # [127, 560], # Abbot (blue)
    [611,  18], # BHTec prox (blue)
    [459, 396], # BHTec med (blue)
    [468, 484], # BHTec longe (blue)
    [146, 224], # BHTec descida (blue)
    
])