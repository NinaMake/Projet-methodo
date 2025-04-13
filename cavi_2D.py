import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,  multivariate_normal


from sklearn.cluster import KMeans

class CAVI:

    # Attention, sigma désigne la variance plutot que sigma^2, et pareil pour s_n.
    def __init__(self, sigma, d, nb_clusters, pi, N):

        # prior parameters
        self.sigma = sigma
        self.nb_clusters = nb_clusters
        self.pi = pi
        self.mu = [np.random.multivariate_normal(
            np.zeros(d), self.sigma*np.identity(d))for k in range(len(self.pi))
                   ]

        # dimensions données
        self.N = N
        self.d = d

        # données
        self.data = np.zeros((N, d))

        # paramètres variationnels
        self.phi = np.zeros((N, nb_clusters))
        # les assignation sont les mêmes pour toutes les coordonnées

        for i in range(self.N):
            self.phi[i, np.random.choice(nb_clusters)] = 1
        # On initialise au hasard les phi (au hasard un cluster)


        self.ELBOS = [1, 2]
       # self.m_0 = np.zeros((nb_clusters))
        self.m_0 = np.array(
            [np.random.multivariate_normal(np.zeros(d), sigma*np.identity(d)
                                           ) for k in range(self.nb_clusters)]
            )

        self.s_0 = np.array(
            [abs(np.random.normal(0, sigma)) for k in range(self.nb_clusters)]
            )

        self.m_n = self.m_0
        self.s_n = self.s_0

    def gmm_rand(self):

        # on tire N fois entier k
        for i in range(self.N):
            mult = np.random.multinomial(1, pi, size = 1)
            k = np.argmax(mult)
            self.data[i, :] = np.random.multivariate_normal(self.mu[k],
                                        np.identity(self.d))


    def init_kmeans(self):
        kmeans_model = KMeans(
            n_clusters=self.nb_clusters, n_init=1, max_iter=100, random_state=1
            )

        # Pas besoin de reshape si self.data est déjà en 2D
        kmeans_model.fit(self.data)

        centroids = kmeans_model.cluster_centers_
        cluster_assignement = kmeans_model.labels_

        self.m_0 = centroids
        K = len(centroids)

        for i in range (N):
            for k in range(K):
                if cluster_assignement[i]== k:
                    self.phi[i,k] = 1
                else :
                    self.phi[i,k] = 0

        data_clusters = [[] for k in range(K)]
        for k in range(K):
            for i in range(N):
                if cluster_assignement[i] == k:
                    data_clusters[k].append(self.data[i])
            self.s_0[k] = np.diag(np.cov(data_clusters[k], rowvar=False))



    def gmm_pdf(self, x):
        res = 0
        for k in range(len(self.pi)):
            res += self.pi[k] * multivariate_normal.pdf(x, self.mu[k],
                                                        np.identity(self.d))
        return res

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
            res += -(self.d * self.s_n[k]**2 + np.dot(self.m_n[k], self.m_n[k])) / (2 * self.sigma**2)
            res -= self.d * np.log(2 * np.pi * self.sigma**2) / 2
            res -= self.d * (1 + np.log(2 * np.pi * self.s_n[k]**2)) / 2

        for i in range(self.N):
            res += -np.log(self.nb_clusters)
            for k in range(self.nb_clusters):
                res += self.phi[i, k] * (
                    -np.dot(self.data[i], self.data[i]) + 2 * np.dot(self.data[i], self.m_n[k]) -
                    (self.d * self.s_n[k]**2 + np.dot(self.m_n[k], self.m_n[k])) -
                    self.d * np.log(2 * np.pi)) / 2
                res += self.phi[i, k] * np.log(self.phi[i, k])
        return res

    def coordinate_ascent(self, nb_iter):
        iter = 0
        while iter < nb_iter and not self.has_converged():
            print(f"m_n: {self.m_n}")

            iter += 1

            for k in range(self.nb_clusters):
                denom = (1 / (self.sigma**2)) + np.sum(self.phi[:, k])
                self.m_n[k] = np.dot(self.phi[:, k], self.data) / denom
                self.s_n[k] = 1 / denom

            for i in range(self.N):
                approx_phi = []
                for k in range(self.nb_clusters):
                    approx_phi.append(np.exp(np.dot(self.m_n[k], self.data[i]) - (
                        self.d * self.s_n[k]**2 + np.dot(self.m_n[k], self.m_n[k])) / 2))
                approx_phi = np.array(approx_phi)
                self.phi[i, :] = approx_phi / np.sum(approx_phi)

            # Calculer l'ELBO
            ELBO = self.calc_elbo()
            self.ELBOS.append(ELBO)

            print(f"Iteration: {iter}")
            print(f"s_n: {self.s_n}")
            print(self.ELBOS[iter])

    def plot_results(self):

        # Définir les limites de la grille en fonction des données
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        xx, yy = xx.ravel(), yy.ravel()
        xy = np.stack([xx, yy], axis=1)

        # Calculer la densité estimée par CAVI pour chaque point de la grille
        pdf = np.zeros(len(xy))
        for i in range(len(xy)):
            for k in range(self.nb_clusters):
                # Densité gaussienne pour le cluster k
                cluster_density = multivariate_normal.pdf(xy[i], mean=self.m_n[k], cov= np.eye(self.d))
                pdf[i] += cluster_density

        # Remodeler la densité pour l'affichage
        pdf = pdf.reshape(100, 100)

        # Afficher les données
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[:, 0], self.data[:, 1], c='0.8', marker='o', alpha=0.3, label="Data")

        # Afficher les contours de la densité estimée
        plt.contour(xx.reshape(100, 100), yy.reshape(100, 100), pdf, alpha = 0.5,colors ='r')

        # Vraie densité
        vraie_pdf = np.zeros(len(xy))
        for i in range(len(xy)):
            vraie_pdf[i] = self.gmm_pdf(xy[i])

        vraie_pdf = vraie_pdf.reshape(100, 100)

        plt.contour(xx.reshape(100, 100), yy.reshape(100, 100), vraie_pdf, colors='g', linestyles = 'dashed')

        # Moyennes estimées
        plt.scatter(self.m_n[:, 0], self.m_n[:, 1], c='red', marker='x', s=100, label="Estimated means")

        plt.xlabel(r"$x_1$", fontsize=14)
        plt.ylabel(r"$x_2$", fontsize=14)
        plt.title("Densities estimated by CAVI", fontsize=16)
        plt.axis('equal')  # Forcer les axes à avoir la même échelle
        plt.legend()
        plt.show()

    def plot_posterior_priore(self):
        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        xx, yy = xx.ravel(), yy.ravel()
        xy = np.stack([xx, yy], axis=1)

        # Calculer la posterior estimée par CAVI pour chaque point de la grille
        pdf_posterior = np.zeros(len(xy))
        for i in range(len(xy)):
            for k in range(self.nb_clusters):
                # Densité gaussienne pour le cluster k
                cluster_density = multivariate_normal.pdf(xy[i], mean=self.m_n[k], cov= self.s_n[k]*np.eye(self.d))
                pdf_posterior[i] += cluster_density

        pdf_posterior = pdf_posterior.reshape(100, 100)

        # Calcul de la prior
        pdf_prior = np.zeros(len(xy))
        for i in range(len(xy)):
            for k in range(self.nb_clusters):
                # Densité gaussienne pour le cluster k
                cluster_density = multivariate_normal.pdf(xy[i], [0, 0], cov= self.sigma*np.eye(self.d))
                pdf_prior[i] += cluster_density

        pdf_prior = pdf_prior.reshape(100, 100)

        # Affichage

        plt.contour(xx.reshape(100, 100), yy.reshape(100, 100), pdf_posterior, cmap="magma")

        plt.contour(xx.reshape(100, 100), yy.reshape(100, 100), pdf_prior, colors ='blue', linestyles= 'dashed')

        plt.scatter(self.m_n[:, 0], self.m_n[:, 1], c='red', marker='x', s=100, label="Estimated means")

        plt.xlabel(r"$x_1$", fontsize=14)
        plt.ylabel(r"$x_2$", fontsize=14)
        plt.title("Comparison of prior and posterior over mu", fontsize=16)
        plt.axis('equal')  # Forcer les axes à avoir la même échelle
        plt.legend()
        plt.show()



sigma = 10  # Écart-type commun
nb_clusters =  3 # Nombre de clusters
pi = np.array([1/3, 1/3, 1/3])  # Proportions des clusters
N = 1000  # Nombre de points de données

np.random.seed(78)
# 40000 close clusters

cavi = CAVI(sigma, 2, nb_clusters, pi, N)

cavi.gmm_rand()

cavi.coordinate_ascent(50)
print(cavi.m_n)
print(cavi.mu)
cavi.plot_results()
cavi.plot_posterior_priore()

plt.figure()
elbo = cavi.ELBOS
del elbo[0:2]

plt.plot (elbo)
plt.ylabel('ELBO')
plt.xlabel('iterations')
plt.show()

# On vérifie qu'on obtient aussi la distribution 1/K (ici 1/3)
print(np.sum(cavi.phi[:,0]), np.sum(cavi.phi[:,1]), np.sum(cavi.phi[:,2]))
print(cavi.mu, cavi.m_n)
