# Dockerfile pour webots.cloud
# Se base sur l'image Webots officielle correspondant à la version
# déclarée dans mon_monde.wbt (R2025a). $WEBOTS_DEFAULT_IMAGE est une
# variable fournie automatiquement par le serveur webots.cloud -- elle
# pointe vers la bonne image Docker Webots sans qu'on ait à écrire le
# nom exact en dur.

FROM $WEBOTS_DEFAULT_IMAGE

# Dépendances Python nécessaires au contrôleur (fly_vision_controller.py) :
# - numpy : calculs du circuit EMD/LPLC2
# - opencv-python-headless : conversion image caméra -> niveaux de gris
#   (version "headless" car pas besoin d'interface graphique OpenCV,
#   seulement du traitement d'image, ce qui évite des dépendances système
#   inutiles dans le conteneur)
RUN pip3 install --no-cache-dir numpy opencv-python-headless
