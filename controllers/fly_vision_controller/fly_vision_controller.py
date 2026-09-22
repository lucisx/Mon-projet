"""
Contrôleur Webots : drone piloté par le circuit EMD -> LPLC2 bio-inspiré
de la mouche, avec entrée caméra réelle (pas de rétine simulée).

A placer dans : mon_projet/controllers/fly_vision_controller/fly_vision_controller.py
Le nom du DOSSIER doit être identique au nom du FICHIER .py (obligatoire
pour que Webots le reconnaisse).

Prérequis dans l'éditeur de scène Webots (.wbt) :
  - Un robot/drone (ex: Mavic2Pro déjà fourni dans les PROTO Webots)
  - Un device Camera nommé "camera" ajouté comme enfant du drone
  - Ce contrôleur assigné au champ "controller" du robot

Dépendances (déjà présentes dans Webots, ou à installer dans son Python) :
    pip install numpy opencv-python-headless --break-system-packages
"""

import numpy as np
import cv2
from controller import Robot, Camera, Motor


# ----------------------------------------------------------------------
# Réutilisation du circuit déjà validé (copie simplifiée ici pour un
# fichier autonome ; sinon tu peux aussi faire
# `from fly_vision_drone_sim import EMDLayer, LPLC2Layer` si ce fichier
# est accessible dans le même dossier / PYTHONPATH).
# ----------------------------------------------------------------------

N_BINS = 64  # résolution panoramique de la "rétine" reconstruite depuis la caméra


class EMDLayer:
    def __init__(self, n=N_BINS, tau_steps=3):
        self.n = n
        self.tau_steps = tau_steps
        self.history = []

    def step(self, intensity):
        self.history.append(intensity.copy())
        if len(self.history) > self.tau_steps + 1:
            self.history.pop(0)
        if len(self.history) <= self.tau_steps:
            return np.zeros(self.n)
        i_now, i_delayed = self.history[-1], self.history[0]
        i_next_now = np.roll(i_now, -1)
        i_next_delayed = np.roll(i_delayed, -1)
        return (i_now * i_next_delayed) - (i_next_now * i_delayed)


class LPLC2Layer:
    def __init__(self, n=N_BINS, receptive_field=6, weights_path=None):
        self.n = n
        self.rf = receptive_field
        idx = np.arange(n)
        center = n // 2

        if weights_path is not None:
            data = np.load(weights_path)
            loaded = data["population_weights"]
            if len(loaded) != n:
                loaded = np.interp(np.linspace(0, len(loaded), n, endpoint=False),
                                    np.arange(len(loaded)), loaded)
            self.population_weights = loaded / loaded.sum()
        else:
            self.population_weights = np.exp(-0.5 * ((idx - center) / (n / 6)) ** 2)
            self.population_weights /= self.population_weights.sum()

        self.lr_weights = np.tanh((idx - center) / (n / 6))

    def step(self, emd_signal):
        n, rf = self.n, self.rf
        responses = np.zeros(n)
        for i in range(n):
            left_idx = (np.arange(i - rf, i) % n)
            right_idx = (np.arange(i + 1, i + rf + 1) % n)
            mid = rf // 2
            near_left = np.mean(np.clip(-emd_signal[left_idx[-mid:]], 0, None))
            far_left = np.mean(np.clip(-emd_signal[left_idx[:mid]], 0, None))
            near_right = np.mean(np.clip(emd_signal[right_idx[:mid]], 0, None))
            far_right = np.mean(np.clip(emd_signal[right_idx[-mid:]], 0, None))
            quadrants = np.array([near_left, far_left, near_right, far_right])
            responses[i] = np.prod(quadrants) ** 0.25 if np.all(quadrants > 0) else 0.0

        expansion_signal = np.sum(responses * self.population_weights)
        turn_bias = np.sum(responses * self.lr_weights) / (np.sum(responses) + 1e-8)
        return expansion_signal, turn_bias


# ----------------------------------------------------------------------
# CONVERSION CAMERA -> RETINE PANORAMIQUE
# La caméra Webots donne une image rectangulaire (largeur x hauteur).
# On la réduit à un vecteur 1D de N_BINS valeurs de luminance moyenne
# par colonne, ce qui joue le rôle des "ommatidies" de la rétine.
# ----------------------------------------------------------------------

def camera_image_to_retina(camera, n_bins=N_BINS):
    width, height = camera.getWidth(), camera.getHeight()
    raw = camera.getImage()  # buffer BGRA brut
    img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # moyenne de luminance par colonne, puis ré-échantillonnage sur n_bins
    col_means = gray.mean(axis=0)  # taille = width
    retina = np.interp(np.linspace(0, width, n_bins, endpoint=False),
                        np.arange(width), col_means)
    return retina / 255.0  # normaliser entre 0 et 1, comme sense_retina()


# ----------------------------------------------------------------------
# BOUCLE PRINCIPALE WEBOTS
# ----------------------------------------------------------------------

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Adapte les noms de moteurs/actionneurs à ton modèle de drone Webots
    # (ex: Mavic2Pro utilise des moteurs "front left propeller", etc. ;
    # pour un drone plus simple/générique, on suppose ici des champs
    # de contrôle haut-niveau via un device "yaw" et "thrust" custom,
    # à adapter selon le PROTO exact que tu utilises).
    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
    for m in motors:
        m.setPosition(float("inf"))  # mode vitesse continue

    emd_layer = EMDLayer()
    lplc2_layer = LPLC2Layer(weights_path="connectome_weights.npz")

    base_speed = 60.0       # vitesse de rotation de base des hélices
    danger_gain = 40.0
    steer_gain = 15.0

    while robot.step(timestep) != -1:
        retina = camera_image_to_retina(camera)
        emd_signal = emd_layer.step(retina)
        expansion_signal, turn_bias = lplc2_layer.step(emd_signal)

        # signal de danger -> réduit la poussée avant, augmente le virage
        speed_adjust = -danger_gain * expansion_signal
        turn_adjust = -steer_gain * turn_bias

        left_speed = base_speed + speed_adjust - turn_adjust
        right_speed = base_speed + speed_adjust + turn_adjust

        front_left_motor.setVelocity(left_speed)
        rear_left_motor.setVelocity(left_speed)
        front_right_motor.setVelocity(right_speed)
        rear_right_motor.setVelocity(right_speed)

        print(f"expansion={expansion_signal:.4f}  turn_bias={turn_bias:.4f}  "
              f"L={left_speed:.1f}  R={right_speed:.1f}")


if __name__ == "__main__":
    main()
