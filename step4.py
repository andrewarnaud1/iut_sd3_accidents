import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


# Importer le fichier time_encoding.csv
victime = pd.read_csv("step3/time_encoding.csv", low_memory=False)

# On extrait du tableau la latitude et la longitude

X_lat = victime["lat"]
X_long = victime["long"]

# On définit tous nos points à classifier

X_cluster = np.array((list(zip(X_lat, X_long))))

# Kmeans nous donne pour chaque point la catégorie associée

clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Enfin on ajoute les catégories dans la base d'entraînement

geo = pd.Series(clustering.labels_)
victime["geo"] = geo

# Exporter les données dans un fichier geo_encoding.csv zipé
victime.to_csv("step4/geo_encoding.csv.zip", index=False, compression="zip")
