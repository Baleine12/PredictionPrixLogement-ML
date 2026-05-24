import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1 Création d'une base simulée

np.random.seed(42)

n = 1500

surface = np.random.randint(20, 140, n)
nombre_pieces = np.random.randint(1, 7, n)
distance_centre = np.random.uniform(0.5, 25, n)
anciennete_batiment = np.random.randint(0, 80, n)
etage = np.random.randint(0, 12, n)
balcon = np.random.choice([0, 1], n, p=[0.65, 0.35])
parking = np.random.choice([0, 1], n, p=[0.55, 0.45])

quartier = np.random.choice(
    ["centre", "residentiel", "peripherie"],
    n,
    p=[0.25, 0.45, 0.30]
)

prix = (
    50000
    + surface * 4200
    + nombre_pieces * 9000
    - distance_centre * 3500
    - anciennete_batiment * 900
    + etage * 1200
    + balcon * 18000
    + parking * 22000
    + np.random.normal(0, 25000, n)
)

prix = np.where(quartier == "centre", prix + 70000, prix)
prix = np.where(quartier == "residentiel", prix + 30000, prix)
prix = np.where(quartier == "peripherie", prix - 15000, prix)

prix = np.maximum(prix, 50000)
prix = np.round(prix, 0)

df = pd.DataFrame({
    "surface": surface,
    "nombre_pieces": nombre_pieces,
    "distance_centre": distance_centre,
    "anciennete_batiment": anciennete_batiment,
    "etage": etage,
    "balcon": balcon,
    "parking": parking,
    "quartier": quartier,
    "prix": prix
})

df.to_csv("donnees/logements_simules.csv", index=False)

print("Aperçu de la base :")
print(df.head())

print("Nombre d'observations :", len(df))

print("Statistiques descriptives :")
print(df.describe())

# 2 Graphiques simples

plt.figure(figsize=(7, 5))
plt.scatter(df["surface"], df["prix"], alpha=0.4)
plt.title("Prix des logements selon la surface")
plt.xlabel("Surface")
plt.ylabel("Prix")
plt.tight_layout()
plt.savefig("sorties/graphiques/prix_selon_surface.png")
plt.close()

plt.figure(figsize=(7, 5))
plt.scatter(df["distance_centre"], df["prix"], alpha=0.4)
plt.title("Prix des logements selon la distance au centre")
plt.xlabel("Distance au centre")
plt.ylabel("Prix")
plt.tight_layout()
plt.savefig("sorties/graphiques/prix_selon_distance.png")
plt.close()

plt.figure(figsize=(7, 5))
df.groupby("quartier")["prix"].mean().sort_values().plot(kind="bar")
plt.title("Prix moyen selon le quartier")
plt.xlabel("Quartier")
plt.ylabel("Prix moyen")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sorties/graphiques/prix_moyen_quartier.png")
plt.close()

# 3 Préparation des variables

df_model = pd.get_dummies(df, columns=["quartier"], drop_first=True)

X = df_model.drop("prix", axis=1)
y = df_model["prix"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# 4 Régression linéaire


modele_lineaire = LinearRegression()
modele_lineaire.fit(X_train, y_train)
pred_lineaire = modele_lineaire.predict(X_test)
mae_lineaire = mean_absolute_error(y_test, pred_lineaire)
rmse_lineaire = np.sqrt(mean_squared_error(y_test, pred_lineaire))
r2_lineaire = r2_score(y_test, pred_lineaire)

print("Régression linéaire :")
print("MAE :", round(mae_lineaire, 2))
print("RMSE :", round(rmse_lineaire, 2))
print("R2 :", round(r2_lineaire, 3))



# 5 Random Forest


modele_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

modele_rf.fit(X_train, y_train)

pred_rf = modele_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)

print("Random Forest :")
print("MAE :", round(mae_rf, 2))
print("RMSE :", round(rmse_rf, 2))
print("R² :", round(r2_rf, 3))



# 6 Comparaison des modèles


resultats = pd.DataFrame({
    "modele": ["Regression lineaire", "Random Forest"],
    "MAE": [mae_lineaire, mae_rf],
    "RMSE": [rmse_lineaire, rmse_rf],
    "R2": [r2_lineaire, r2_rf]
})

print("Comparaison des modèles :")
print(resultats)

resultats.to_csv("sorties/comparaison_modeles.csv", index=False)


# 7 Importance des variables


importance = pd.DataFrame({
    "variable": X.columns,
    "importance": modele_rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("Importance des variables :")
print(importance)

importance.to_csv("sorties/importance_variables.csv", index=False)

plt.figure(figsize=(8, 5))
plt.barh(importance["variable"], importance["importance"])
plt.title("Importance des variables - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("sorties/graphiques/importance_variables.png")
plt.close()



# 8. Prix réels vs prix prédits


plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred_rf, alpha=0.5)
plt.title("Prix réels et prix prédits - Random Forest")
plt.xlabel("Prix réel")
plt.ylabel("Prix prédit")
plt.tight_layout()
plt.savefig("sorties/graphiques/prix_reels_vs_predits.png")
plt.close()



# 9. Export des prédictions


predictions = pd.DataFrame({
    "prix_reel": y_test,
    "prix_predit_rf": pred_rf,
    "erreur": y_test - pred_rf
})

predictions.to_csv("sorties/predictions_prix.csv", index=False)

print("Projet terminé avec succès.")