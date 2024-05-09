#!/usr/bin/env python3
import sys

from GMPHD_Filter import GMPHDFilter
from GaussianMixture import GaussianMixture
from Simulation import Simulation
import numpy as np


def define_gmphd_model(model_number):
    """
    :param model_number: definira gmphd filtar za pojedinu situaciju
    :return: gmphd filtar
    """
    survaillance_region = np.array([[-1000, 1000], [-1000, 1000]])  # raspon x-a, raspon y-a
    p_s = 0.99  # vjerojatnost preživljavanja mete
    p_d = 0.98  # vjerojatnost detekcije mete
    t_s = 1  # vrijeme izmedu uzoraka
    # Koristimo za podrezivanje
    T = 1e-5  # granica za podrezivanje
    U = 4  # granica za spajanje
    Jmax = 100  # maksimalan dozvoljen broj Gaussovih mješavina

    if model_number == "1":
        lambda_c = 12.5 * 1e-6  # prosjecan broj mjerenja prouzrokovanih šumom po m^2
        sigma_v = 5  # standardna devijacija šuma procesa
        sigma_r = 10  # standardna devijacija šuma mjerenja
        birth_w = [0.1, 0.1]  # ocekivan broj meta iz m_y,k (odgovarajući w[i] za birth_m[i])
        birth_m = [np.array([250, 250, 0, 0]), np.array([-250, -250, 0, 0])]  # srednje vrijednosti m_y,k
        birth_P = [np.diag([100, 100, 25, 25]),
                   np.diag([100, 100, 25, 25])]  # odgovarajuća matrica kovarijance P_y,k za m_y,k
        birth_GM = GaussianMixture(birth_w, birth_m, birth_P)
    elif model_number == "2":
        lambda_c = 5 * 1e-6  # prosjecan broj mjerenja prouzrokovanih šumom po m^2
        sigma_v = 10  # standardna devijacija šuma procesa
        sigma_r = 10  # standardna devijacija šuma mjerenja
        birth_w = [0.1, 0.1, 0.1, 0.1, 0.1]  # ocekivan broj meta iz m_y,k (odgovarajući w[i] za birth_m[i])
        birth_m = [np.array([0, 900, 0, 0]), np.array([600, 300, 0, 0]),
                   np.array([-600, -600, 0, 0]), np.array([-600, 0, 0, 0]),
                   np.array([-600, 600, 0, 0])]  # srednje vrijednosti m_y,k
        birth_P = [np.diag([100, 100, 25, 25]), np.diag([100, 100, 25, 25]),
                   np.diag([100, 100, 25, 25]), np.diag([100, 100, 25, 25]),
                   np.diag([100, 100, 25, 25])]  # odgovarajuća matrica kovarijance P_y,k za m_y,k
        birth_GM = GaussianMixture(birth_w, birth_m, birth_P)
    else:
        lambda_c = 1e-5  # prosjecan broj mjerenja prouzrokovanih šumom po m^2
        sigma_v = 5  # standardna devijacija šuma procesa
        sigma_r = 10  # standardna devijacija šuma mjerenja
        birth_w = [0.1, 0.1, 0.1, 0.1]  # ocekivan broj meta iz m_y,k (odgovarajući w[i] za birth_m[i])
        birth_m = [np.array([900, 0, 0, 0]), np.array([900, -900, 0, 0]), np.array([-900, -900, 0, 0]), np.array([-900, 0, 0, 0])]  # srednje vrijednosti m_y,k
        birth_P = [np.diag([100, 100, 25, 25]), np.diag([100, 100, 25, 25]), np.diag([100, 100, 25, 25]), np.diag([100, 100, 25, 25])]  # odgovarajuća matrica kovarijance P_y,k za m_y,k
        birth_GM = GaussianMixture(birth_w, birth_m, birth_P)

    return GMPHDFilter(survaillance_region, lambda_c, t_s, p_s, p_d, sigma_v, sigma_r, T, U, Jmax, birth_GM)


def define_simulation(gmphd_filter, simulation_num):
    """
    :param gmphd_filter: gmphd filtar definiran za pojedinu simulaciju
    :param simulation_num: broj simulacije, može biti 1, 2 ili 3
    :return: vraća simulaciju s određenim postavkama
    """
    x_range = gmphd_filter.surveillance_region[0][1] - gmphd_filter.surveillance_region[0][0]
    y_range = gmphd_filter.surveillance_region[1][1] - gmphd_filter.surveillance_region[1][0]
    region = int(np.round(x_range * y_range))
    avg_clutter_returns = int(round(gmphd_filter.lambda_c * region))

    if simulation_num == "1":
        time_birth_measurement = {1: [np.array([-250, -250, 12, -7.5]), np.array([250, 250, 2.5, -12])],
                                  66: [np.array([400, -500, -5.5, -2.5])]}
    elif simulation_num == "2":
        time_birth_measurement = {1: [np.array([0, 900, 0, -17]), np.array([600, 300, -12, -6]),
                                      np.array([-600, -600, 12, 12]), np.array([-600, 0, 12, 0]),
                                      np.array([-600, 600, 12, -12]), np.array([300, 700, -9, -10])]}
    else:
        time_birth_measurement = {1: [np.array([900, 0, -9, 9]), np.array([-900, 0, 9, 9]), np.array([900, 0, -9, -9]),
                                      np.array([900, 0, -13, 0]), np.array([900, -900, -18, 18]),
                                      np.array([-900, -900, 5, 15])],
                                  25: [np.array([-900, -900, 0, 10])],
                                  50: [np.array([0, 0, -10, -10])]}

    return Simulation(gmphd_filter.H, gmphd_filter.R, avg_clutter_returns, time_birth_measurement, simulation_num, gmphd_filter.t_s)


if __name__ == '__main__':
    simulation_number = input("Upišite broj simulacije: ")  # putem komandne linije se prima broj simulacije, simulacija može biti 1, 2 ili 3
    gmphd_filter = define_gmphd_model(simulation_number)
    simulation = define_simulation(gmphd_filter, simulation_number)
    target_trajectories = simulation.generate_trajectories()
    clutter_measurements, target_measurements, Z = simulation.generate_measurements(target_trajectories)
    w = []
    m = []
    P = []
    gaussian_mixture = GaussianMixture(w, m, P)
    X_k = []
    old_time = 0.0

    cnt = 0
    for time, z in Z.items():
        gaussian_mixture = gmphd_filter.prediction(gaussian_mixture)
        ni, K, P_k, S_k = gmphd_filter.construction(gaussian_mixture)
        gaussian_mixture = gmphd_filter.update(gaussian_mixture, z, ni, K, P_k, S_k)
        gaussian_mixture = gmphd_filter.prune(gaussian_mixture)
        X_k.append(gmphd_filter.state_extraction(gaussian_mixture))

    simulation.plot_graphs(target_trajectories, Z, X_k, target_measurements)
