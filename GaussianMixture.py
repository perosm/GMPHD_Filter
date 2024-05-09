class GaussianMixture:
    def __init__(self, w, m, P):
        """
        Gaussova mješavina
        :param w: lista parametara težine tj. očekivan broj meta koji proizlazi iz odgovarajuće srednje vrijednosti m
        :param m: lista srednjih vrijednosti
        :param P: matrica kovarijance, određuje raspršenost intenziteta u blizi odgovarajuće srednje vrijednosti m
        """
        self.w = w
        self.m = m
        self.P = P
