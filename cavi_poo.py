import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import seaborn as sns

# tester les graines suivantes : 124, 1981, 22051965, 31031965


class CAVI:
    def __init__(self, sigma, nb_clusters, pi, N):

        # prior parameters
        self.sigma = sigma
        self.nb_clusters = nb_clusters
        self.pi = pi
        self.mu = [np.random.normal(0, self.sigma)for k in range(nb_clusters)]

        # dimensions données
        self.N = N

        # données
        self.data = np.zeros((N))

        # paramètres variationnels

        self.ELBOS = [1, 2]

        # On initialise par défaut avec la priore, ou bien on appelle
        # la méthode kmeans

        self.m_0 = np.array(
            [np.random.normal(0, sigma) for k in range(self.nb_clusters)
             ])
        self.s_0 = np.array(
            [abs(np.random.normal(0, sigma)) for k in range(self.nb_clusters)]
            )

        self.phi = np.zeros((N, nb_clusters))
        for i in range(self.N):
            self.phi[i, np.random.choice(nb_clusters)] = 1
        # On met au hasard les individus dans un cluster


        self.m_n = self.m_0

        self.s_n = self.s_0

    def gmm_rand(self):

        # on génére les données en fonction du cluster dans lequel est chaque individu

        for i in range(self.N):
            mult = np.random.multinomial(1, pi, size = 1)
            k = np.argmax(mult)
            self.data[i] = np.random.normal(self.mu[k],
                                            1)

    def gmm_pdf(self, x):
        res = 0
        for k in range(len(self.pi)):
            res += norm.pdf(x, self.mu[k], 1)
           # self.pi[k] * on l'enlève pour normaliser
        return res

    def init_kmeans(self):
        kmeans_model = KMeans(n_clusters=self.nb_clusters,
                              n_init=1, max_iter=100,
                              random_state=1)

        data_init = self.data.reshape((self.N, 1))

        kmeans_model.fit(data_init)
        centroids, cluster_assignement = kmeans_model.cluster_centers_, kmeans_model.labels_

        # On initialise les paramètres variationnels avec ce qu'on a trouvé
        centroids = centroids.reshape((self.m_0.shape))
        cavi.m_0 = centroids
        K = len(centroids)

        for i in range(N):
            for k in range(K):
                if cluster_assignement[i] == k:
                    self.phi[i, k] = 1
                else:
                    self.phi[i, k] = 0

        # Initialisation de la covariance avec les covariances empiriques

        data_clusters = [[] for k in range(K)]
        for k in range(K):
            for i in range(N):
                if cluster_assignement[i] == k:
                    data_clusters[k].append(self.data[i])
                    # attention data_clusters liste de liste, pas tableau numpy
            self.s_0[k] = np.cov(data_clusters[k], rowvar=False)

    def has_converged(self):
        diff = abs(self.ELBOS[-1] - self.ELBOS[-2])
        return diff < 0.01

    # def facteurs_variationnels(self):
    #     res = 1
    #     for i in range(self.N):
    #         for k in range(self.nb_clusters):
    #             res *= norm.pdf(self.mu[k], self.m_n[k], self.s_n[k])
    #             res *= self.phi[i, k]

    def calc_elbo(self):
        res = 0

        for k in range(self.nb_clusters):
            res += -(self.s_n[k]**2 + self.m_n[k]**2
                     )/2*self.sigma**2 - (1/2)*np.log(2*np.pi*self.sigma**2) - (
                         1+np.log(2*np.pi*self.s_n[k]**2))/2

        for i in range(self.N):
            res += -np.log(self.nb_clusters)
            for k in range(self.nb_clusters):
                res += self.phi[i, k]*(
                    -self.data[i]**2 + 2*self.data[i]*self.m_n[k]-(
                        self.s_n[k]**2 + self.m_n[k]**2) - np.log(2*np.pi))/2
                res += self.phi[i, k]*np.log(self.phi[i, k])
        return res

    def coordinate_ascent(self, nb_iter):
        iter = 0
        while iter < nb_iter and not self.has_converged():
            for k in range(self.nb_clusters):
                denom = 1/(self.sigma) + np.sum(self.phi[:, k])
                self.m_n[k] = np.dot(self.phi[:, k], self.data)/denom
                self.s_n[k] = 1/denom

            for i in range(self.N):
                approx_phi = []
                for k in range(self.nb_clusters):
                    approx_phi.append(np.exp(self.m_n[k]*self.data[i] - (
                        self.s_n[k] + self.m_n[k]**2)/2))
                for k in range(self.nb_clusters):
                    self.phi[i, k] = approx_phi[k]/np.sum(np.array(approx_phi))

            # calcul ELBO
            ELBO = self.calc_elbo()
            self.ELBOS.append(ELBO)

            iter += 1

            print(f"Iteration: {iter}")
            print(f"m_n: {self.m_n}")
            print(f"s_n: {self.s_n}")
            print(self.ELBOS[iter])

    def plot_results(self):
        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(-20, 20, 1000)
        ax.plot(xx, self.gmm_pdf(xx))
        # Histogramme des données
        a = self.data
        ax.hist(a, density=True, bins=30)

        for k in range(self.nb_clusters):
            vals = np.random.normal(self.m_n[k], 1, size=1000)
            sns.kdeplot(vals,  color='k', ax=ax)
        plt.show()


sigma = 10  # écart_type priore
nb_clusters = 2  # Nombre de clusters
pi = np.array([0.5, 0.5])  # Proportions des clusters
N = 1000  # Nombre de points de données

np.random.seed(124)

cavi = CAVI(sigma, nb_clusters, pi, N)
cavi.gmm_rand()
print(cavi.mu)
print(cavi.m_0)
print(np.sum(cavi.phi[:, 1]))

cavi.coordinate_ascent(50)
cavi.plot_results()

plt.figure()
elbo = cavi.ELBOS
del elbo[0:2]
print(elbo)


# # # On vérifie que l'ELBO augmente (bon... pas toujours le cas) : pb observé
# pour graine 1981
plt.plot(elbo)
plt.ylabel('ELBO')
plt.xlabel('itérations')
plt.show()
