import pandas as pd


# Exporter dans ce dossier le dataframe avec l'encodage de hrmn, du mois dans un fichier appelé geo_encoding.csv zipé
# Importer le fichier geo_encoding.csv zipé
victime = pd.read_csv("step4/geo_encoding.csv.zip", low_memory=False)

print(victime.columns)

y = victime["grav"]

features = [
    "catu",
    "sexe",
    "trajet",
    "secu",
    "catv",
    "an_nais",
    "mois",
    "occutc",
    "obs",
    "obsm",
    "choc",
    "manv",
    "lum",
    "agg",
    "int",
    "atm",
    "col",
    "gps",
    "catr",
    "circ",
    "vosp",
    "prof",
    "plan",
    "surf",
    "infra",
    "situ",
    "hrmn",
    "geo",
]

X_train_data = pd.get_dummies(victime[features].astype(str))

# Rajouter la colonne grav dans le dataframe X_train_data
X_train_data["grav"] = y

# On exporte les données dans un fichier csv
X_train_data.to_csv("step5/one_hot_encoding.csv", index=False)
