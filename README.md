# OCR-P9-recommandations

## Contenu du dépôt :

### Recommendation_CB-CF.ipynb :

Notebook de développement des algorithmes Content-Based et Collaborative Filtering

### Script :

Script d'entraînement du modèle : charge les fichiers clicks.csv depuis le container Azure "Input_data", réalise l'entraînement du modèle et sauvegarde les matrices résultantes dans le container "trained-model".

### Recommender :

API serverless déployée sur Azure pour interagir avec l'app mobile. Elle reçoit en entrée un n° d'utilisateur et renvoie une liste des 5 recommandations les plus pertinentes.

### Mobile app

Application mobile. Identique à l'application originale hormis le fichier config.json qui a été modifié avec l'ajout de l'URL de l'API serverless.