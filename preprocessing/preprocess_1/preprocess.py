# http://pbpython.com/categorical-encoding.html
import pandas as pd
from sklearn import preprocessing


def stringToLabel():
    traindata = pd.read_csv("dataset/train.csv", sep=";", index_col=0)
    testdata = pd.read_csv("dataset/test.csv", index_col=0)

    #Fazer isso para todos os que precisam
    #Arrancar fora todos os que foram o segundo atributo de algo
    cleanup_nums = {"MSZoning":    {"C (all)": 0, "RM": 1, "RH": 2, "RL": 3, "FV": 4},
                    "Street":       {"Grvl": 0, "Pave": 1},
                    "Alley":        {"Grvl": 0, "Pave": 1},
                    # Terreno inclinado é mais caro para construir, talvez a ordem deverá ser mudada
                    "LotShape":     {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3}, 
                    "LandContour":  {"Bnk": 0, "Lvl": 1, "Low": 2, "HLS": 3},
                    "Utilities":    {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
                    "LotConfig":    {"Inside": 0, "FR2": 1, "Corner": 2, "FR3": 3, "CulDSac": 4},
                    # Poderia ser invertido
                    "LandSlope":    {"Sev": 0, "Mod": 1, "Gtl": 2},
                    "Neighborhood": {"MeadowV": 0, "IDOTRR": 1, "BrDale": 2, "BrkSide": 3, "Edwards": 4, "OldTown": 5, "Sawyer": 6, "Blueste": 7, "SWISU": 8, "NPkVill": 9, "NAmes": 10, "Mitchel": 11, "SawyerW": 12, "NWAmes": 13, "Gilbert": 14, "Blmngtn": 15, "CollgCr": 16, "Crawfor": 17, "ClearCr": 18, "Somerst": 19, "Veenker": 20, "Timber": 21, "StoneBr": 22, "NridgHt": 23, "NoRidge": 24},
                    "Condition1":   {"Artery": 0, "RRAe": 1, "Feedr": 2, "RRAn": 3, "Norm": 4, "RRNe": 5, "RRNn": 6, "PosN": 7, "PosA": 8},
                    #RRNe foi adicionado, pois não temos ele na tabela. -> Arrancar fora
                    "Condition2":   {"RRNn": 0, "Artery": 1, "Feedr": 2, "RRAn": 3, "Norm": 4, "RRAe": 5, "RRNe": 6, "PosN": 7, "PosA": 8},
                    "BldgType":     {"2fmCon": 0, "Duplex": 1, "Twnhs": 2, "TwnhsE": 3, "1Fam": 4},
                    "HouseStyle":   {"1.5Unf": 0, "SFoyer": 1, "1.5Fin": 2, "2.5Unf": 3, "SLvl": 4, "1Story": 5, "2Story": 6, "2.5Fin": 7},
                    "RoofStyle":    {"Gambrel": 0, "Gable": 1, "Mansard": 2, "Flat": 3, "Hip": 4, "Shed": 5},
                    "RoofMatl":     {"Roll": 0, "ClyTile": 1, "CompShg": 2, "Metal": 3, "Tar&Grv": 4, "WdShake": 5, "Membran": 6, "WdShngl": 7},
                    "Exterior1st":  {"BrkComm": 0, "AsphShn": 1, "CBlock": 2, "AsbShng": 3, "MetalSd": 4, "Wd Sdng": 5, "WdShing": 6, "Stucco": 7, "HdBoard": 8, "Plywood": 9, "BrkFace": 10, "VinylSd": 11, "CemntBd": 12, "Stone": 13, "ImStucc": 14},
                    #Exterior2nd
                    "MasVnrType":   {"BrkCmn": 0, "None": 1, "BrkFace": 2, "Stone": 3},
                    "ExterQual":    {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                    "ExterCond":    {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                    "Foundation":   {"Slab": 0, "BrkTil": 1, "CBlock": 2, "Stone": 3, "Wood": 4, "PConc": 5},
                    "BsmtQual":     {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                    "BsmtCond":     {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                    "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
                    "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6 },
                    "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6 },
                    "Heating":      {"Floor": 0, "Grav": 1, "Wall": 2, "OthW": 3, "GasW": 4, "GasA": 5},
                    "HeatingQC":    {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                    "CentralAir":   {"N": 0, "Y": 1},
                    "Electrical":   {"NA": 0, "Mix": 1, "FuseP": 2, "FuseF": 3, "FuseA": 4, "SBrkr": 5}, 
                    "KitchenQual":   {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                    "Functional":   {"Maj2": 0, "Sev": 1, "Min2": 2, "Min1": 3, "Maj1": 4, "Mod": 5, "Typ": 6},
                    "FireplaceQu":  {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                    "GarageType":   {"NA": 0, "CarPort": 1, "Detchd": 2, "2Types": 3, "Basment": 4, "Attchd": 5, "BuiltIn": 6},
                    "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
                    "GarageQual":   {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                    "GarageCond":   {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                    "PavedDrive":   {"N": 0, "P": 1, "Y": 2},
                    "PoolQC":       {"NA": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
                    "Fence":        {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
                    #MiscFeature
                    #SaleType
                    #SaleCondition
                    }

    lines_drop = ["Exterior2nd", "MiscFeature", "MiscVal", "SaleType", "SaleCondition"]
    traindata.drop(lines_drop, axis=1, inplace=True)
    testdata.drop(lines_drop, axis=1, inplace=True)

    traindata.fillna(0, inplace=True)
    testdata.fillna(0, inplace=True)

    traindata.replace(cleanup_nums, inplace=True)
    testdata.replace(cleanup_nums, inplace=True)

    traindata.to_csv("dataset/train_2.csv", sep=";", index_col=0)
    testdata.to_csv("dataset/test_2.csv", sep=";", index_col=0)

    return traindata, testdata