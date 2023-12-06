import pandas as pd


# Read the data from the CSV file
victime = pd.read_csv("step2/missing_values_deleted.csv", low_memory=False)

hrmn = pd.cut(victime["hrmn"], 24, labels=[str(i) for i in range(24)])

victime["hrmn"] = hrmn.values

# Exporter dans ce dossier le dataframe avec l'encodage de hrmn, du mois dans un fichier appelé time_encoding.csv
victime.to_csv("step3/time_encoding.csv", index=False)
