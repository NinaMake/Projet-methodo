import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import seaborn as sns
import timeit

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
        self.m_0 = centroids
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
            iter += 1
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



            print(f"Iteration: {iter}")
            print(f"m_n: {self.m_n}")
            print(f"s_n: {self.s_n}")
            print(self.ELBOS[iter])

    # def plot_results(self):
    #     fig, ax = plt.subplots(1, 1)
    #     xx = np.linspace(-20, 20, 1000)
    #     ax.plot(xx, self.gmm_pdf(xx))
    #     # Histogramme des données
    #     a = self.data
    #     ax.hist(a, density=True, bins=30)

    #     for k in range(self.nb_clusters):
    #         vals = np.random.normal(self.m_n[k], 1, size=1000)
    #         sns.kdeplot(vals,  color='k', ax=ax)
    #     plt.show()

    def plot_results(self):
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Histogramme des données
        sns.histplot(self.data, bins=30, kde=False, stat="density", color="skyblue", alpha=0.5, label="Données")

        # Courbe de la densité GMM (sans pondération pour être comparable au KDE)
        xx = np.linspace(min(self.data)-3, max(self.data)+3, 1000)
        plt.plot(xx, self.gmm_pdf(xx), label="GMM générative", color='black', linestyle='--', linewidth=2)

        # Courbes KDE des distributions a posteriori de chaque cluster
        for k in range(self.nb_clusters):
            samples = np.random.normal(self.m_n[k], 1, size=1000)
            sns.kdeplot(samples, label=f"Post. Cluster {k+1}", linewidth=2)

        plt.title("Estimation de la densité par CAVI", fontsize=16)
        plt.xlabel("Valeur", fontsize=14)
        plt.ylabel("Densité", fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_posteriore_priore(self):
        plt.figure(figsize=(10, 6))
        xx = np.linspace(-10, 10, 1000)

        # Prior : somme de normales centrées en 0 avec variance sigma
        prior_density = np.zeros_like(xx)
        for k in range(self.nb_clusters):
            prior_density += norm.pdf(xx, loc=0, scale=self.sigma)

        # Posterior : somme de normales centrées sur m_n avec s_n
        posterior_density = np.zeros_like(xx)
        for k in range(self.nb_clusters):
            posterior_density += norm.pdf(xx, loc=self.m_n[k], scale=self.s_n[k])

        plt.plot(xx, prior_density, label="Prior", linestyle='--', color='blue')
        plt.plot(xx, posterior_density, label="Posterior", linestyle='-', color='red')

        plt.title(" Prior comparison with Posterior", fontsize=16)
        plt.xlabel(r"$\mu$", fontsize=14)
        plt.ylabel("Densité", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



sigma = 10 # écart_type priore
nb_clusters = 2  # Nombre de clusters
pi = np.array([0.5, 0.5])  # Proportions des clusters
N = 1000  # Nombre de points de données

np.random.seed(1981)

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

plt.plot(elbo)
plt.ylabel('ELBO')
plt.xlabel('itérations')
plt.show()

cavi.plot_posteriore_priore()
