import numpy as np
from GaussianMixture import GaussianMixture

def gaussian_pdf(z, ni, S_k):
    S_k_inv = np.linalg.inv(S_k)
    numerator = np.exp(-1/2 * np.linalg.multi_dot([z - ni, S_k_inv, np.transpose(z - ni)]))
    denominator = 1 / np.sqrt(np.power((2 * np.pi), 4) * np.linalg.det(S_k))

    return numerator * denominator


class GMPHDFilter:
    def __init__(self, survaillance_region, lambda_c, t_s, p_s, p_d, sigma_v, sigma_r, T, U, Jmax, birth_GM):
        self.surveillance_region = survaillance_region  # raspon x i y koordinata
        self.lambda_c = lambda_c
        self.birth_GM = birth_GM  # Gaussova mjesavina rodenih meta
        self.t_s = t_s  # vrijeme izmedu dva mjerenja
        self.p_s = p_s  # vjerojatnost prezivljavanja mete p_S,k
        self.p_d = p_d  # vjerojatnost detekcije mete p_D,k
        self.sigma_r = sigma_r  # varijanca mjerne nesigurnosti
        self.sigma_v = sigma_v  # varijanca suma procesa
        self.F = np.array([[1, 0, t_s, 0],  # matrica tranzicije stanja F
                           [0, 1, 0, t_s],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = np.array([[(t_s**4) / 4, 0, (t_s**3) / 2, 0],  # matrica suma procesa Q
                           [0, (t_s**4) / 4, 0, (t_s**3) / 2],
                           [(t_s**3) / 2, 0, (t_s**2), 0],
                           [0, (t_s**3) / 2, 0, (t_s**2)]]).dot(sigma_v**2)
        self.H = np.array([[1, 0, 0, 0],  # matrica observacije H
                           [0, 1, 0, 0]])
        self.R = np.array([[1, 0],  # matrica mjerne nesigurnosti R
                           [0, 1]]) * (sigma_r**2)
        self.T = T  # granica za podrezivanje
        self.U = U  # granica za spajanje
        self.Jmax = Jmax  # maksimalan dozvoljen broj Gaussovih mješavina

    def clutter_density_function(self, z):
        """
        :param z: pojedino mjerenje oblika np.ndarray([x, y])
        :return: prosječan broj mjerenja dobivenih od strane šuma po m^2
        """
        x_length = self.surveillance_region[0, 1] - self.surveillance_region[0, 0]
        y_length = self.surveillance_region[1, 1] - self.surveillance_region[1, 0]
        surveillance_area = x_length * y_length

        return self.lambda_c * surveillance_area * (1 / surveillance_area)  # k(z) = lambda_c * V * u(z)

    def update_time(self, delta_t):
        """
        :param delta_t: vremenska razlika između dobivenih mjerenja
        :return:
        """
        self.F = np.array([[1, 0, delta_t, 0],  # matrica tranzicije stanja F
                           [0, 1, 0, delta_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.Q = np.array([[(delta_t**4) / 4, 0, (delta_t**3) / 2, 0],  # matrica suma procesa Q
                           [0, (delta_t**4) / 4, 0, (delta_t**3) / 2],
                           [(delta_t**3) / 2, 0, (delta_t**2), 0],
                           [0, (delta_t**3) / 2, 0, (delta_t**2)]]).dot(self.sigma_v**2)

    def prediction(self, gaussian_mixture):
        """
        korak 1: predikcija rođenja objekata
        korak 2: predikcija za postojeće objekte
        :param gaussian_mixture: Gaussova mješavina prethodnog trenutka
        :return: Gaussova mješavina rođenih i već postojećih objekata
        """
        w_existing = []
        m_existing = []
        P_existing = []

        for w in gaussian_mixture.w:
            w_existing.append(self.p_s * w)  # w_k|k-1 = p_S,k * w_k-1
        for m in gaussian_mixture.m:
            m_existing.append(np.matmul(self.F, m))  # m_k|k-1 = F_k-1 * m_k-1
        for P in gaussian_mixture.P:
            P_existing.append(self.Q + np.linalg.multi_dot([self.F, P, self.F.transpose()]))  # P_k|k-1 = Q_k-1 + F_k-1*P_k-1*F_k-1.T

        return GaussianMixture(self.birth_GM.w + w_existing,
                               self.birth_GM.m + m_existing,
                               self.birth_GM.P + P_existing)

    def construction(self, gaussian_mixture):
        """
        korak 3: konstrukcija PHD komponenti
        :param gaussian_mixture: Gaussova mješavina
        :return:
            ni - koordinate pretpostavljene pozicije objekta
            K - Kalmanov dobitak
            P_k - matrica kovarijance nesigurnosti trenutka k
            S_k - nazivnik Kalmanovog dobitka
        """
        ni = []
        K = []
        S_k = []
        P_k = []
        for m in gaussian_mixture.m:
            ni.append(np.matmul(self.H, m))  # ni_k|k-1 = H_k * m_k|k-1

        for P in gaussian_mixture.P:
            S = self.R + np.linalg.multi_dot([self.H, P, self.H.transpose()])  # S_k = R_k + H_k * P_k|k-1 * H_k.T
            S_inv = np.linalg.inv(S)
            S_k.append(S)
            K_k = np.linalg.multi_dot([P, self.H.transpose(), S_inv])  # K_K = P_k|k-1 * H_k.T * S_k.inv
            K.append(K_k)
            P_k.append(np.matmul(np.identity(4) - np.matmul(K_k, self.H), P))

        return ni, K, P_k, S_k

    def update(self, gm, Z, ni, K, P_k, S_k):
        """
        step 4: update
        :param gm: Gaussove mješavina koraka predikcije
        :param Z: lista mjerenja trenutka k
        :param ni: koordinate pretpostavljene pozicije objekta
        :param K: Kalmanov dobitak
        :param P_k: matrica kovarijance trenutka k
        :param S_k: nazivnik Kalmanovog dobitka
        :return: Gaussova mješavina trenutka k
        """
        w_update = [(1 - self.p_d) * w for w in gm.w]
        m_update = gm.m
        P_update = gm.P

        for z in Z:
            tmp = []
            for j in range(len(gm.w)):
                tmp.append(self.p_d * gm.w[j] * gaussian_pdf(z, ni[j], S_k[j]))
                m_update.append(gm.m[j] + np.matmul(K[j], z - ni[j]))
                P_update.append(P_k[j])

            clutter_density = self.clutter_density_function(z)
            weights_sum = sum(tmp)
            for j in range(len(gm.w)):
                w_update.append(tmp[j] / (clutter_density + weights_sum))

        return GaussianMixture(w_update, m_update, P_update)

    def prune(self, gaussian_mixture):
        """
        Podrezivanje komponenti Gaussove mješavine
        :param gaussian_mixture: Gaussova mješavina trenutka k
        :return: Gaussova mješavina nakon podrezivanja
        """
        w = gaussian_mixture.w
        m = gaussian_mixture.m
        P = gaussian_mixture.P
        I = [i for i in range(len(w)) if gaussian_mixture.w[i] > self.T]

        w_tilda = []
        m_tilda = []
        P_tilda = []
        l = 0

        while len(I) != 0:
            j = I[0]
            L = []

            for i in I:
                if w[i] > w[j]:  # argmax w[i]_k, i je element od I
                    j = i

            for i in I:
                diff = np.linalg.multi_dot([m[i] - m[j], np.linalg.inv(P[i]), np.transpose(m[i] - m[j])])
                if diff <= self.U:  # ako je diff <= U onda mjerenja vjerojatno dolaze od iste mete (merging threshold)
                    L.append(i)  # dodajemo

            w_l = 0.0
            m_l = np.array([0, 0, 0, 0])
            P_l = np.zeros((4, 4))

            for i in L:
                w_l = w_l + w[i]
            for i in L:
                m_l = m_l + (w[i] * m[i])
            m_l = m_l / w_l

            for i in L:
                P_l = P_l + (w[i] * (P[i] + np.outer(m_l - m[i], np.transpose(m_l - m[i]))))
            P_l = P_l / w_l

            w_tilda.append(w_l)
            m_tilda.append(m_l)
            P_tilda.append(P_l)

            I = np.setdiff1d(I, L)  # vraća  vrijednosti iz I koje nisu u L
            l = l + 1

        if l > self.Jmax:
            indexes = np.argsort(w_tilda)[-self.Jmax:]  # vraća redne brojeve vektora koji bi sortirali vektor
            w_tilda = [w_tilda[i] for i in indexes]
            m_tilda = [m_tilda[i] for i in indexes]
            P_tilda = [P_tilda[i] for i in indexes]

        return GaussianMixture(w_tilda, m_tilda, P_tilda)

    def state_extraction(self, gaussian_mixture):
        """
        :param gaussian_mixture: Gaussova mješavina nakon podrezivanja
        :return: lista ekstrahiranih stanja objekata čije težine su veće od 0.5
        """
        w = gaussian_mixture.w
        m = gaussian_mixture.m

        tmp = []
        for i in range(len(w)):
            if w[i] > 0.5:
                for j in range(int(np.round(w[i]))):
                    tmp.append(m[i])

        return tmp
