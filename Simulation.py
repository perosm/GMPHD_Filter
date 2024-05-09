import math

import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    def __init__(self, H, R,  avg_clutter_returns, time_birth_measurement, simulation_num, t_s):
        self.avg_clutter_returns = avg_clutter_returns
        self.survaillance_area = np.array([[-1000, 1000], [-1000, 1000]])
        self.sampling_period = t_s
        self.sampling_time = 100
        self.time_birth_measurement = time_birth_measurement
        self.H = H
        self.R = R
        self.simulation_num = simulation_num
        self.steering_rate = (math.pi / 2) / 100  # konstantni kut zakretanja
        self.alpha = (math.pi / 2) / 100

    def generate_trajectories(self):
        """
        Generiranje putanja objekata s obzirom na početno mjesto pojavljivanja i brzinu objekta
        :return: dictionary putanja objekata oblika: redni broj mete -> np.array([pozicije i brzine mete])
        """
        targets = {}
        target_cnt = 0
        for i in range(1, self.sampling_time):  # za svaki period uzorkovanja
            for t_c in range(target_cnt):  # generiramo točku
                last_measurement = targets[t_c][len(targets[t_c]) - 1]
                speed_x = last_measurement[2]
                speed_y = last_measurement[3]
                new_pos_x = last_measurement[0] + self.sampling_period * last_measurement[2]
                new_pos_y = last_measurement[1] + self.sampling_period * last_measurement[3]
                targets[t_c].append(np.array([new_pos_x, new_pos_y, speed_x, speed_y]))

            if i in self.time_birth_measurement.keys():  # ukoliko se u danom trenutku rodila meta/mete
                for birth_measurement in self.time_birth_measurement[i]:  # iteriramo po svim metama/meti
                    targets[target_cnt] = [birth_measurement]  # dodajemo početne vrijednosti prilikom rođenja
                    target_cnt += 1

        return targets

    def generate_measurements(self, trajectories):
        """
        Generiranje mjerenja iz trajektorija
        :param trajectories: dictionary putanja objekata oblika: redni broj mete -> np.array([pozicije i brzine mete])
        :return:
            clutter_measurements - mjerenja dobivena od strane šuma
            target_measurements - mjerenja dobivena od strane objekata
            measurements - clutter_measurements i target_measurements zajedno
        """
        if self.simulation_num == "3":
            trajectories = self.add_steering_to_a_trajectory(trajectories, [0, 1])
        clutter_measurements = self.generate_clutter()
        target_measurements = self.generate_target_measurements(trajectories)
        measurements = self.combine_measurements(clutter_measurements, target_measurements)

        return clutter_measurements, target_measurements, measurements

    def add_steering_to_a_trajectory(self, trajectories, which_trajectories):
        """
        :param trajectories: dictionary putanja objekata oblika: redni broj mete -> np.array([pozicije i brzine mete])
        :param which_trajectories: lista rednih brojeva trajektorija kojima želimo promijeniti oblik putanje
        :return: dictionary putanja objekata oblika: redni broj mete -> np.array([pozicije i brzine mete])
        """
        for i in which_trajectories:
            tmp = []
            for j in range(len(trajectories[i])):
                v_x = trajectories[i][j][2]
                v_y = trajectories[i][j][3]
                if i % 2 == 0:
                    x = abs(trajectories[i][len(trajectories[i]) - 1][0] - trajectories[i][0][0]) * math.cos(self.alpha)
                    y = abs(trajectories[i][len(trajectories[i]) - 1][1] - trajectories[i][0][1]) * math.sin(self.alpha)
                else:
                    x = abs(trajectories[i][len(trajectories[i]) - 1][0] - trajectories[i][0][0]) * (- math.cos(self.alpha))
                    y = abs(trajectories[i][len(trajectories[i]) - 1][1] - trajectories[i][0][1]) * math.sin(self.alpha)
                tmp.append(np.array([x, y, v_x, v_y]))
                self.alpha += self.steering_rate  # ažuriranje kuta
            trajectories[i] = tmp
            self.alpha = self.steering_rate

        return trajectories

    def generate_clutter(self):
        """
        Generiranje mjerenja od strane šuma
        :return: dictionary oblika: vrijeme -> šumovita mjerenja
        """
        clutter = {}

        for i in range(self.sampling_time):
            c = []
            #  Poissonova RFS -> broj šumovitih mjerenja uzorkujemo kao Poissonovu slučajnu varijablu
            #                 -> uzorke odnosno šumovita mjerenja uzorkujemo iz uniformne distribucije
            for j in range(np.random.poisson(lam=self.avg_clutter_returns)):
                clutter_x = np.random.uniform(self.survaillance_area[0][0], self.survaillance_area[0][1])
                clutter_y = np.random.uniform(self.survaillance_area[1][0], self.survaillance_area[1][1])
                c.append(np.array([clutter_x, clutter_y]))
            clutter[i] = c

        return clutter

    def generate_target_measurements(self, trajectories):
        """
        :param trajectories: putanje svih objekata
        :return: mjerenja dobivena od strane objekata
        """
        target_measurements = {i: [] for i in range(self.sampling_time)}
        for trajectory in trajectories.keys():
            for i in range(len(trajectories[trajectory])):  # mjerenja uzorkujemo kao normalnu slučajnu varijablu
                measurement = np.random.multivariate_normal(np.matmul(self.H, trajectories[trajectory][i]), self.R)
                target_measurements[self.sampling_time - len(trajectories[trajectory]) + i].append(measurement)

        return target_measurements

    def combine_measurements(self, clutter, target_measurements):
        """
        :param clutter: mjerenja generirana od strane šuma
        :param target_measurements: mjerenja generirana od strane objekata
        :return: mjerenja generirana od strane šuma i do strane objekata zajedno
        """
        measurements = {i: [] for i in range(self.sampling_time)}
        for i in range(self.sampling_time):
            measurements[i] = target_measurements[i] + clutter[i]

        return measurements

    def extract_position_and_time(self, target_measurements):
        """
        Potrebno za grafički prikaz putanja
        :param target_measurements: mjerenja od strane jednog objekta
        :return: lista tuple-a (x, y, vrijeme)
        """
        measurements = []
        targets_num = len(target_measurements)  # u koliko vremenskih trenutaka je objekt generirao mjerenja
        starting_time = self.sampling_time - targets_num  # ukupno vrijeme mjerenja - broj vremenskih trenutaka u kojima je generirao mjerenja objekta
        time_cnt = starting_time

        for target_measurement in target_measurements:
            measurements.append((target_measurement[0], target_measurement[1], time_cnt))
            time_cnt += 1

        return measurements

    def extract_from_estimates(self, X):
        """
        :param X: ekstrahirane estimacije stanja objekata GMPHD filtra
        :return: listu tuple-a (x, y, vrijeme) za pojedinu estimaciju objekta GMPHD filtra
        """
        time_and_coordinates = []
        time = 0
        for i in range(self.sampling_time):
            if len(X[i]) != 0:
                for coord in X[i]:
                    time_and_coordinates.append((coord[0], coord[1], time))
            time += 1

        return time_and_coordinates

    def extract_time_coordinate(self, Z, which_coordinate):
        """
        Potrebno za grafički prikaz
        :param Z: dictionary oblika vrijeme -> mjerenja
        :param which_coordinate: redni broj koordinate; 0 označava x, 1 označava y
        :return: lista parova (vrijeme, vrijednost koordinate)
        """
        time_coordinate = []
        for time in range(len(Z)):
            for z in Z[time]:
                time_coordinate.append((time, z[which_coordinate]))

        return time_coordinate

    def plot_graphs(self, target_trajectories, Z, X, target_measurements):

        ###  Graf [0][0]  ###
        target_position_in_time = []
        for target in target_trajectories.keys():
            target_position_in_time.append(self.extract_position_and_time(target_trajectories[target]))

        first_graph = []
        for tpit in target_position_in_time:
            tmp = []
            for x, y, time in tpit:
                tmp.append((x, y))
            first_graph.append(tmp)

        colors = ["red", "green", "blue", "purple", "orange", "olive", "brown", "cyan"]
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))
        fig.tight_layout(pad=5.0)
        #axs[0][0].set_title(label="Putanje objekata; 'o': mjesta rođenja objekata, 'x' mjesta umiranja objekata")
        axs[0][0].set_xlim(self.survaillance_area[0][0], self.survaillance_area[0][1])
        axs[0][0].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[0][0].set_xlabel('x koordinata (m)')
        axs[0][0].set_ylabel('y koordinata (m)')

        for i in range(len(first_graph)):
            axs[0][0].scatter(*first_graph[i][0], marker='o', color=colors[i])
            axs[0][0].plot(*zip(*first_graph[i]), color=colors[i])
            axs[0][0].scatter(*first_graph[i][len(first_graph[i]) - 1], marker='x', color=colors[i])


        ###  Graf [1][0]  ###
        #axs[1][0].set_title(label="Mjerenja i prave pozicije objekata")
        axs[1][0].set_xlim(0, self.sampling_time)
        axs[1][0].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[1][0].set_xlabel('vrijeme (s)')
        axs[1][0].set_ylabel('x koordinata (m)')

        time_coordinate_x = self.extract_time_coordinate(Z, 0)
        time, coordinate_x = zip(*time_coordinate_x)
        sc10 = axs[1][0].scatter(time, coordinate_x, marker='x', alpha=0.4, label='Measurements')
        object_measurement = self.extract_time_coordinate(target_measurements, 0)
        time, coordinate_x = zip(*object_measurement)
        sc10_black = axs[1][0].scatter(time, coordinate_x, marker='x', alpha=0.4, label='Measurements', color='black')
        trajectories_time_x = []
        for tpit in target_position_in_time:
            tmp = []
            for x, _, time in tpit:
                tmp.append((time, x))
            trajectories_time_x.append(tmp)

        tr10 = [axs[1][0].plot(*zip(*trajectories_time_x[i]), color=colors[i])[0] for i in range(len(trajectories_time_x))]
        #axs[1][0].legend([sc10] + [sc10_black] + tr10, ['Measurements'] + ['Object originated measurements'] + ['Target ' + str(i) for i in range(1, len(tr10) + 1)], loc='upper left')
        axs[1][0].legend([sc10] + [sc10_black], ['Mjerenja'] + ['Mjerenja od strane objekata'], loc='upper left')


        ###  Graf [2][0]  ###
        #axs[2][0].set_title(label="Mjerenja i prave pozicije objekata.")
        axs[2][0].set_xlim(0, self.sampling_time)
        axs[2][0].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[2][0].set_xlabel('vrijeme (s)')
        axs[2][0].set_ylabel('y koordinata (m)')

        time_coordinate_y = self.extract_time_coordinate(Z, 1)
        time, coordinate_y = zip(*time_coordinate_y)
        sc20 = axs[2][0].scatter(time, coordinate_y, marker='x', alpha=0.4)

        object_measurement = self.extract_time_coordinate(target_measurements, 1)
        time, coordinate_y = zip(*object_measurement)
        sc20_black = axs[2][0].scatter(time, coordinate_y, marker='x', alpha=0.4, label='Measurements', color='black')

        trajectories_time_y = []
        for tpit in target_position_in_time:
            tmp = []
            for _, y, time in tpit:
                tmp.append((time, y))
            trajectories_time_y.append(tmp)

        tr20 = [axs[2][0].plot(*zip(*trajectories_time_y[i]), color=colors[i])[0] for i in range(len(trajectories_time_y))]
        # axs[2][0].legend([sc20] + tr20, ['Measurements'] + ['Target ' + str(i) for i in range(1, len(tr20) + 1)], loc='upper left')
        axs[2][0].legend([sc20] + [sc20_black], ['Mjerenja'] + ['Mjerenja od strane objekta'], loc='upper left')
        estimate_coords_time = self.extract_from_estimates(X)


        ### Graf [0][1] ###
        estimate_coords = []
        for x, y, _ in estimate_coords_time:
            estimate_coords.append((x, y))
        axs[0][1].set_title(label="Position estimates of the Gaussian mixture PHD filter.")
        axs[0][1].set_xlim(self.survaillance_area[0][0], self.survaillance_area[0][1])
        axs[0][1].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[0][1].set_xlabel('x koordinata (m)')
        axs[0][1].set_ylabel('y koordinata (m)')

        for i in range(len(first_graph)):
            axs[0][1].scatter(*first_graph[i][0], marker='o', color=colors[i])
            axs[0][1].plot(*zip(*first_graph[i]), color=colors[i])
            axs[0][1].scatter(*first_graph[i][len(first_graph[i]) - 1], marker='x', color=colors[i])

        axs[0][1].scatter(*zip(*estimate_coords_time), marker='o', edgecolor='black', color='white')


        ### Graf [1][1] ###
        #axs[1][1].set_title(label="Position estimates of the Gaussian mixture PHD filter.")
        axs[1][1].set_xlim(0, self.sampling_time)
        axs[1][1].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[1][1].set_xlabel('vrijeme (s)')
        axs[1][1].set_ylabel('x koordinata (m)')
        tr11 = [axs[1][1].plot(*zip(*trajectories_time_x[i]), color=colors[i])[0] for i in range(len(trajectories_time_x))]
        estimate_time_x = []
        for x, _, time in estimate_coords_time:
            estimate_time_x.append((time, x))

        sc11 = axs[1][1].scatter(*zip(*estimate_time_x), marker='o', edgecolor='black', color='white')

        # axs[1][1].legend([sc11] + tr11, ['PHD filter estimates'] + ['Target ' + str(i) for i in range(1, len(tr11) + 1)], loc='upper left')
        axs[1][1].legend([sc11], ['estimacije GMPHD Filtra'], loc='upper left')


        ### Graf [2][1] ###
        #axs[2][1].set_title(label="Position estimates of the Gaussian mixture PHD filter.")
        axs[2][1].set_xlim(0, self.sampling_time)
        axs[2][1].set_ylim(self.survaillance_area[1][0], self.survaillance_area[1][1])
        axs[2][1].set_xlabel('vrijeme (s)')
        axs[2][1].set_ylabel('y koordinata (m)')
        tr21 = [axs[2][1].plot(*zip(*trajectories_time_y[i]), color=colors[i])[0] for i in range(len(trajectories_time_x))]

        estimate_time_y = []
        for _, y, time in estimate_coords_time:
            estimate_time_y.append((time, y))

        sc21 = axs[2][1].scatter(*zip(*estimate_time_y), marker='o', edgecolor='black', color='white')
        # axs[2][1].legend([sc21] + tr21, ['PHD filter estimates'] + ['Target ' + str(i) for i in range(1, len(tr21) + 1)], loc='upper left')
        axs[2][1].legend([sc21], ['estimacije GMPHD Filtra'], loc='upper left')
        plt.show()
