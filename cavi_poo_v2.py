import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns


class CAVI:
    def __init__(self, sigma, nb_clusters, pi, N):

        # prior parameters
        self.sigma = sigma
        self.nb_clusters = nb_clusters
        self.pi = pi
        self.mu = [np.random.normal(0, self.sigma)for k in range(len(self.pi))]

        # dimensions données
        self.N = N

        # données
        self.data = np.zeros((N))

        # paramètres variationnels
        self.phi = np.zeros((N, nb_clusters))
        self.ELBOS = [1, 2]
       # self.m_0 = np.zeros((nb_clusters))
        self.m_0 = [np.random.normal(0, sigma) for k in range(self.nb_clusters)]
        self.s_0 = [abs(np.random.normal(0, sigma)) for k in range(self.nb_clusters)]

        self.m_n = self.m_0
       # self.s_0 = np.ones((nb_clusters))
        self.s_n = self.s_0

    def gmm_rand(self):

        # on tire N fois entier k
        for i in range(self.N):
            mult = np.random.multinomial(1, self.pi, size=1)
            k = np.argmax(mult)
            self.data[i] = np.random.normal(self.mu[k],
                                            1)

    def gmm_pdf(self, x):
        res = 0
        for k in range(len(self.pi)):
            res += self.pi[k] * norm.pdf(x, self.mu[k], 1)
        return res

    def has_converged(self):
        diff = abs(self.ELBOS[-1] - self.ELBOS[-2])
        return diff < 0.01

    def facteurs_variationnels(self):
        res = 1
        for i in range(self.N):
            for k in range(self.nb_clusters):
                res *= norm.pdf(self.mu[k], self.m_n[k], self.s_n[k])
                res *= self.phi[i, k]

    def calc_elbo(self):
        res = 0
        for k in range(self.nb_clusters):
            res += -(self.s_n[k]**2 + self.m_n[k]**2
                     )/self.sigma**2 - 1/2*np.log(2*np.pi*self.sigma**2) - (
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
            print(f"m_n: {self.m_n}")
            iter += 1

            for i in range(self.N):
                approx_phi = []
                for k in range(self.nb_clusters):
                    approx_phi.append(np.exp(self.m_n[k]*self.data[i] - (
                        self.s_n[k] + self.m_n[k]**2)/2))
                for k in range(self.nb_clusters):
                    self.phi[i, k] = approx_phi[k]/np.sum(np.array(approx_phi))

            for k in range(self.nb_clusters):
                denom = 1/(self.sigma) + np.sum(self.phi[:, k])
                self.m_n[k] = np.dot(self.phi[:, k], self.data)/denom
                self.s_n[k] = 1/denom

            # calcul ELBO
            ELBO = self.calc_elbo()
            self.ELBOS.append(ELBO)

            print(f"Iteration: {iter}")

            print(f"s_n: {self.s_n}")
            print(self.ELBOS[iter])

    def plot_results(self):
        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(-20, 20, 1000)
        ax.plot(xx, self.gmm_pdf(xx))
        # Histogramme des données
        a = self.data
        ax.hist(a, density=True, bins='auto')

        for k in range(self.nb_clusters):
            vals = np.random.normal(self.m_n[k], 1, size=1000)
            sns.kdeplot(vals,  color='k', ax=ax)
        plt.show()

    def plot(self):
        # Vraie densité
        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(-20, 20, 1000)
        ax.plot(xx, self.gmm_pdf(xx))

        # Histogramme des données
        a = self.data
        ax.hist(a, density=True, bins='auto')
        plt.show()


sigma = 10  # Écart-type commun
nb_clusters = 2  # Nombre de clusters
pi = np.array([0.5, 0.5])  # Proportions des clusters
N = 1000  # Nombre de points de données

cavi = CAVI(sigma, nb_clusters, pi, N)
cavi.gmm_rand()
cavi.coordinate_ascent(50)
print(cavi.m_n)
print(cavi.mu)
cavi.plot_results()

plt.figure()
elbo = cavi.ELBOS
print(elbo)
del elbo[0:2]
print(elbo)


#Pour vérifier que l'algorithme est correcte : tracer la fonction de coût
# Descend avec le nombre d'itérations, atteint un plateau
plt.plot (elbo)
plt.ylabel('ELBO')
plt.xlabel('itérations')
plt.show()
