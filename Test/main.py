import numpy as np
import random as rd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

fichier = open("Donnees.txt", 'a')
t0 = time.time()
Instances_ss_distanciation = []
fichier.write('Instances_ss_distanciation : \n')
iteration_test = 10
for iter in range(iteration_test):
    print(iter)
    #### INITIALISATION
    if (True):
        affiche_temps = False
        affichage = False
        frequence_affichage = 1
        list_departement = ["IMI","GCC","SGEF","VET","GMM","GI"]
        Nb_indiv = 800
        pop = np.zeros((3, 16))
        pop[0] = np.array([0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0.00, 0.00, 0.00, 0.00]) / 100
        pop[1] = np.floor(Nb_indiv * pop[0])
        erreur = Nb_indiv - sum(pop[1])
        pop[1, 5] = pop[1, 5] + erreur
        pop[2] = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

        ## mise à jour de Nb_indiv
        Nb_indiv = sum(pop[1])
        ## Tableau contenant les vitesses souhaitées de chaque classe d'age
        Taille = 16
        vitess_m = np.zeros((2, Taille))
        vitess_m[0] = [1.10, 1.57, 1.78, 1.99, 1.89, 1.84, 1.81, 1.81, 1.75, 1.71, 1.67, 1.65, 1.65, 1.63, 1.07, 0.07]
        vitess_m[1] = [0.292, 0.292, 0.292, 0.323, 0.323, 0.246, 0.246, 0.246, 0.246, 0.246, 0.243, 0.243, 0.243, 0.255,
                          0.255, 0.255]
        ## Tableau contenant les vitesses souhaitées de chaque classe d'age
        Taille = 16
        dist_soc = np.zeros((2, Taille))
        dist_soc[0] = [1.20, 1.20, 1.26, 1.26, 1.35, 1.35, 1.34, 1.34, 1.35, 1.35, 1.29, 1.29, 1.33, 1.33, 1.34, 1.34]
        dist_soc[1] = [0.415, 0.415, 0.415, 0.415, 0.305, 0.305, 0.305, 0.305, 0.305, 0.305, 0.397, 0.397, 0.397, 0.397,
                          0.397, 0.397]

        ## Tableau contenant les rayons de chaque classe d'age
        Taille = 16
        Masse_m = np.zeros((2, Taille))
        Masse_m[0] = [9.79, 21.129, 34.525, 53.456, 58.08, 64.32, 68.4, 71.5, 59.1, 65.3, 58.99, 60.36, 72.1, 71.995,
                         67.005, 70.39]
        Masse_m[1] = [6.1, 2.816, 6.553, 8.798, 13.1, 13.1, 10.3, 10.3, 10.8, 10.8, 10.13, 10.3, 12.02, 13.63, 15.5,
                         16.71]
        densite = 500
        ##Tableau contenant les rayons
        Rayon_m = np.sqrt(Masse_m / (500 * np.pi))
        ## Tableau contenant l'amplitude de la force de repulsion pour differentes classes d'age
        Taille = 16
        A_m = np.zeros((2, Taille))
        A_m[0] = 700 * np.ones((1, Taille))
        A_m[1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ## Tableau contenant la portee de la force de repulsion pour chaque classe
        Taille = 16
        B_m = np.zeros((2, Taille))
        B_m[0] = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        B_m[1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ## Tableau contenant le tau de relaxation de chauqe classe
        Taille = 16
        Tau_m = np.zeros((2, Taille))
        Tau_m[0] = [0.5, 0.5, 0.5, 0.54, 0.54, 0.54, 0.54, 0.54, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71]
        Tau_m[1] = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ## duree de visite du centre pour chaque classe
        Taille = 16
        T_m = np.zeros((2, Taille))
        T_m[0] = [300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300];
        ##Les stands
        Stands=[]
        ## la durï¿½e de simulation, le pas de temps espace et le nombre d'itérations
        T = 5*60 # cinq minutes
        h = 0.5;  # pas du temps 0.1 seconde
        N_iter = int(T / h)
        n = 0
        ## le domaine dans lequel va ï¿½voluer le systï¿½me
        Xmin = -1
        Xmax = 111.
        Ymin = -1.
        Ymax = 105.
        ## pour la marche alï¿½atoire
        Direcs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]])
        ## Pour Fast Marching
        ph = 1
        # Lorsque la sortie est en bordure d'espace, augmenter la taille de l'espace d'ï¿½volution pour ne pas avoir de problï¿½me
        nph = 5
        Xminn = Xmin - nph * ph
        Xmaxn = Xmax + nph * ph
        Yminn = Ymin - nph * ph
        Ymaxn = Ymax + nph * ph
        ## Les parametres intervenant dans les forces
        ## la force repulsive pieton-pieton
        A_obj = []
        B_obj = []
        alpha = 0.9
        d_obj = []
        R_obj = 10
        ## la force repulsive pieton-obstacle
        A_obs = 1000
        B_obs = 0.8
        d_obs = 1.
        #######################"
        ## Creation de la geometrie
        eps = 1
        ## Création des objets
        Masse = []
        Objets = []
        Rayon = []
        Ray = []
        Allure = []
        Tau = []
        Code = []
        Departement = []
        Obstacles_Rectangle = []
        Stand=[]
        Zone_Interdite = []
        ## Matrices pour la transmission du virus
        p_infecte = []
        p_infectious = []
        Temps = []
        nbr_s = []
        nb_stand_infecte = [0]
        nbr_infecte = [0]
        nbr_indirecte = [0]
        nbr_directe = [0]
        ## Parmetres des equations de Concentration du virus
        sigma = (0.1732 / 3600);  # duree de vie du virus sur les surfaces
        d0 = 2;  # distance maximale à laquelle une surface est contaminee
        pd = 0.04
        hx = 0.5;  # le pas de discretisation
        d_conta = 1.5
        ## discretisation de l'espace
        x = np.arange(Xminn, Xmaxn+hx, hx)
        y = np.arange(Yminn, Ymaxn+hx, hx)
        Nx = len(x)
        Ny = len(y)
        ## creation du maillage
        [X, Y] = np.meshgrid(x, y)
        # Initialisation des matrices de taux de production et de concentration
        C = np.zeros((len(X),len(X[0])))

        n_infecte = 0;  # nombre de personnes infectees
        n_directe = 0;  # le nombre de transmission directe
        n_indirecte = 0;  # le nombre de transmission indirecte
        n_stand_infecte = 0;
        # Initialisation de la matrice contenant les indices des individus
        p_infectious = []
        ##matrice contenant le nombre de contamination pour chaque classe d'age
        TpsStandMin,TpsStandMax = 30, 60 # 1 minute de stand en moyenne
        TpsStandMin,TpsStandMax = int(TpsStandMin / h), int(TpsStandMax/h)
        StandCap = 10

    #### FONCTIONS ANNEXES
    def direction_souhaitee(n,D_S):
        Nb_objets = len(Objets)
        if (n == 1):
            choice = np.random.randint(0, 8, size = Nb_objets)
            D_S = np.array([Direcs[choix] for choix in choice])
        if (n % 10 == 0):
            aa = np.random.randint(0, Nb_objets, size = int(np.floor(Nb_objets * 0.5)))
            choice = np.random.randint(0, 8, size = int(np.floor(Nb_objets * 0.5)))
            D_S[aa] = Direcs[choice]
        return D_S

    def Dis_obstacle_rectangle(j):
        A = []
        Objet = q[j]
        rayon = Rayon[j]
        for obstacle in Obstacles_Rectangle:
            x1,y1,x2,y2 = obstacle
            dy = ( (-1*y2 - rayon + Objet[1])*(y2 < Objet[1]) + ( - y1 + rayon + Objet[1])*(y1 > Objet[1]) ) * (Objet[0] > x1 and Objet[0] < x2) + 10e99 * (Objet[0] <= x1 or Objet[0] >= x2)
            dx = ( ( Objet[0] - rayon - x2)*(Objet[0] > x2) + ( - x1 + rayon + Objet[0])*(Objet[0] < x1) ) * (Objet[1] < y2 and Objet[1] > y1) + 10e99 * (Objet[1] <= y1 or Objet[1] >= y2)
            d = np.sqrt((dx < 10000)*dx**2 + (dy < 10000)*dy**2)
            A.append([(d_obs > abs(dx))*A_obs*np.exp((d_obs - d)/B_obs)*((-1)*(dx <= 0)+ (dx > 0)),(d_obs > abs(dy))*A_obs*np.exp((d_obs - d)/B_obs)*((-1)*(dy<=0) + (dy > 0)) ])
        return np.array(A)

    def Dis_objet_objet(i):
        A = []
        Objet = q[i]
        rayon = Rayon[i]
        dir = D_S[i]
        for j in range(len(Objets)):
            x2,y2 = Objet
            x1,y1 = q[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - rayon - Rayon[j]
            dist = max(0,dist)
            if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) != 0:
                e = [(x1 - x2)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2),(y1 - y2)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2)]
            else:
                e = [0,0]
            phi = -1*dir[0]*e[0] - dir[1]*e[1]
            coef = (dist < R_obj)*(alpha + (1. - alpha)/2.*(1.+np.cos(phi)))*np.array(A_obj[j])*np.exp((d_obj[j] - dist)/B_obj[j])
            A.append( [coef*e[0],coef*e[1]] )
            return np.array(A)

    def Ajouter_Obstacle_Rectangle(x,y,longueur,largeur):
        Obstacles_Rectangle.append([x,y,x+longueur,y+largeur])

    def Ajouter_Zone_Interdite(x,y,largeur,longueur):
        Zone_Interdite.append([x,y,x+largeur,y+longueur])

    def Ajouter_Objet(x,y,ray,cod,allure):
        Objets.append([x,y])
        Rayon.append(ray)
        Masse.append([densite*np.pi*ray**2,densite*np.pi*ray**2])
        Code.append(cod)
        Allure.append(allure)
        d_obj.append(dist_soc[0,cod]+ dist_soc[1,cod]*rd.random())
        A_obj.append(A_m[0,cod]+ A_m[1,cod]*rd.random())
        B_obj.append(B_m[0,cod]+ B_m[1,cod]*rd.random())
        tau= Tau_m[0,cod] + Tau_m[1,cod]*rd.gauss(0,1)
        Tau.append([tau,tau])
        Departement.append(list_departement[rd.randint(0,5)])

    def Ajouter_Plusieurs_Objets(N,cod):
        global Mass
        if(N > 0):
            for i in range(N):
                neglige = 1
                Mass = Masse_m[0, cod] + Masse_m[1, cod] * rd.gauss(0,1)
                while (Mass < 0):
                    Mass = Masse_m[0, cod] + Masse_m[1, cod] * rd.gauss(0,1)
                rayon = np.sqrt(Mass / (500 * np.pi))
                while (neglige == 1):
                    rx = rd.random()
                    ry = rd.random()
                    q2 = Xminn + rx * (Xmaxn - Xminn),Yminn + ry * (Ymaxn - Yminn)
                    vitess = vitess_m[0, cod] + vitess_m[1, cod] * rd.gauss(0,1)
                    while (vitess < 0):
                        vitess = vitess_m[0, cod] + vitess_m[1, cod] * rd.gauss(0,1)
                    Distance_mur_sup = Ymax - q2[1] - rayon
                    Distance_mur_inf = q2[1] - Ymin - rayon
                    Distance_mur_dro = Xmax - q2[0] - rayon
                    Distance_mur_gau = q2[0] - Xmin - rayon
                    interieur_cadre = (Distance_mur_sup > 0) and (Distance_mur_inf > 0) and (Distance_mur_dro > 0) and (Distance_mur_gau > 0)
                    interieur_obstacle = 1
                    interieur_objet = 1
                    interieur_zone = 1
                    for zone in Zone_Interdite:
                        if (q2[0] > zone[0] - rayon and q2[0] < zone[2] + rayon and q2[1] > zone[1] - rayon and q2[1] < zone[3] + rayon):
                            interieur_zone *= 0
                    for obstacle in Obstacles_Rectangle:
                        if (q2[0] > obstacle[0] - rayon and q2[0] < obstacle[2] + rayon and q2[1] > obstacle[1] - rayon and q2[1] < obstacle[3] + rayon):
                            interieur_obstacle *= 0
                    for objet in Objets:
                        if (q2[0] > objet[0] - 2 * rayon and q2[0] < objet[0] + 2 * rayon and q2[1] > objet[1] - 2 * rayon and q2[1] < objet[1] + 2 * rayon):
                            interieur_objet *= 0
                    if (interieur_cadre and interieur_objet == 1 and interieur_obstacle == 1 and interieur_zone == 1):
                        neglige = 0
                        Ajouter_Objet(q2[0], q2[1], rayon, cod, vitess)
                    else:
                        neglige = 1

    def in_cadre(pos):
        x = pos[0]
        y = pos[1]
        return (x > Xmin and x < Xmax and y > Ymin and y < Ymax and (x < 60 or y < 60 ))


    def in_zone(k,i):
        x = q[k][0]
        y = q[k][1]
        X,Y,l,h,dep=Stands[i]
        if X+l/5<x<X+4*l/5 and Y+h/5<y<Y+4*h/5 and Departement[k] == dep:
            return True
        else:
            return False

    def zone(k):
        resu=-1
        for i in range(len(Stands)):
            if in_zone(k,i):
                resu=i
        return resu

    #### AJOUT OBSTACLES ET OBJETS
    if (True):
        Ajouter_Zone_Interdite(60,60,50+eps,45+eps)
        #bas
        Ajouter_Obstacle_Rectangle(Xmin,Ymin,Xmax-Xmin,eps)
        #droite bas
        Ajouter_Obstacle_Rectangle(Xmax-eps,Ymin,eps,60+eps)
        #gauche
        Ajouter_Obstacle_Rectangle(Xmin,Ymin,eps,Ymax-Ymin)
        #haut
        Ajouter_Obstacle_Rectangle(Xmin,Ymax-eps,60+eps,eps)
        #obstacle haut droite
        Ajouter_Obstacle_Rectangle(60,60,50+eps,eps)
        Ajouter_Obstacle_Rectangle(60,60,eps,45)
        Ajouter_Obstacle_Rectangle(0,45,17,11)
        Ajouter_Obstacle_Rectangle(0,34,25,11)
        Ajouter_Obstacle_Rectangle(0,10,8,25)
        #stands
        mini=1
        for i in range(3):
            for j in range(18):
                Stands.append([6+48*j/18,89+3*i,1,1])
            Stands.append([30,79-9*i,7,5])
        Stands.append([28,61,9,5])
        Stands.append([28,52,9,5])
        Stands.append([32,43,8,5])
        Stands.append([31,25,7,5])
        Stands.append([31,16,7,5])
        Stands.append([18,79,4.5,2.5])
        Stands.append([42,79,6,5])
        Stands.append([42,70,7,5])
        Stands.append([42,61,7,5])
        Stands.append([43,52,6,5])
        Stands.append([46,43,9,5])
        Stands.append([18,70,4.5,5])
        Stands.append([13.5,61,9,5])
        Stands.append([61,43,7,5])
        Stands.append([23,16,4.5,5])
        Stands.append([23,25,4.5,5])
        Stands.append([42,16,4.5,5])
        Stands.append([42,25,4.5,5])
        Stands.append([60,10,15,10])
        Stands.append([60,25,10,10])
        Stands.append([46,34,7,5])
        Stands.append([38,34,4,5])
        Stands.append([50,16,4.5,5])
        Stands.append([50,25,4.5,5])
        #ajout des personnes
        for i in range(len(pop[2])):
            Ajouter_Plusieurs_Objets(int(pop[1,i]),i)
        q = np.array(Objets)
        p_infectious = rd.sample([i for i in range(len(Objets))],3)
        ### Initialisation de la matrice contenant les indices des susceptibles
        p_s = np.zeros((len(Objets)))
        for p in p_infectious:
            p_s[p] = 1
        D_S = np.array(range(len(Objets)),dtype= tuple)
        for stand in Stands:
            stand.append(list_departement[rd.randint(0,5)])

    ## Stand contaminé (direct)
    n_st_dir = 3
    list_st_dir = []
    for i in range(n_st_dir):
        r = rd.randint(54,len(Stands) - 1)
        while (r in list_st_dir):
            r = rd.randint(54, len(Stands) - 1)
        list_st_dir.append(r)

    StandNbr = [0 for i in range(len(Stands))]
    dimZ=len(Stands)+1
    Z=[[ rd.randint(TpsStandMin,TpsStandMax)*(Departement[j] == Stands[i][4]) for i in range(len(Stands))] for j in range(len(Objets))]
    Z0 = [ np.array(Z[i]).copy() for i in range(len(Z))]
    V_ant = np.zeros((len(Objets),2))
    fig, ax = plt.subplots()


    while (n < N_iter + 1 and len(Objets) > 0):
        t0 = time.time()
        Temps.append(n)
        if (n%100 == 0):
            print("iteration inter simulation : ", n)
        if (n%frequence_affichage == 0 and affichage):
            ax.cla()
            ax.scatter(0, 0, s=0.1, c='black')
            for j in range(len(q)):
                ax.add_patch(plt.Circle((q[j, 0], q[j, 1]), Rayon[j], color="white"))
            for j in p_infectious:
                ax.add_patch(plt.Circle((q[j, 0], q[j, 1]), Rayon[j], color="red"))
            for j in p_infecte:
                ax.add_patch(plt.Circle((q[j, 0], q[j, 1]), Rayon[j], color="yellow"))
            for obstacle in Obstacles_Rectangle:
                x = obstacle[0]
                y = obstacle[1]
                width = obstacle[2] - obstacle[0]
                height = obstacle[3] - obstacle[1]
                ax.add_patch(patches.Rectangle((x, y), width, height, color="black"))
            for stand in Stands:
                ax.add_patch(patches.Rectangle((stand[0],stand[1]),0.2,stand[3],color="green"))
                ax.add_patch(patches.Rectangle((stand[0], stand[1]), stand[2], 0.2, color="green"))
                ax.add_patch(patches.Rectangle((stand[0]+stand[2], stand[1]), 0.2, stand[3], color="green"))
                ax.add_patch(patches.Rectangle((stand[0], stand[1]+stand[3]), stand[2], 0.2, color="green"))
            for ind_stand in list_st_dir:
                stand = Stands[ind_stand]
                ax.add_patch(patches.Rectangle((stand[0],stand[1]),0.2,stand[3],color="red"))
                ax.add_patch(patches.Rectangle((stand[0], stand[1]), stand[2], 0.2, color="red"))
                ax.add_patch(patches.Rectangle((stand[0]+stand[2], stand[1]), 0.2, stand[3], color="red"))
                ax.add_patch(patches.Rectangle((stand[0], stand[1]+stand[3]), stand[2], 0.2, color="red"))
            ax.set(title = "n = {0}".format(n), xlim = (Xminn,Xmaxn),ylim = (Yminn, Ymaxn))
            im = ax.imshow(C,origin = "lower", extent = (Xminn,Xmaxn,Yminn,Ymaxn),vmin = 0, vmax = 0.01)
            right_inset_ax = fig.add_axes([.55, .6, .2, .2])
            right_inset_ax.plot(Temps,nbr_infecte,'r')
            right_inset_ax.plot(Temps, nbr_indirecte,'b')
            right_inset_ax.plot(Temps, nbr_directe,'g')
            right_inset_ax.plot(Temps, nb_stand_infecte, 'y')
            right_inset_ax.set(xticks = [i*(i%500 == 0) for i in range(n)], yticks = [i*(i%10 == 0) for i in range(n_infecte+1)])
            plt.pause(0.001)
        n = n+1
        if (affiche_temps):
            print(time.time() - t0, "affichage")
        t0 = time.time()
        Objets = np.array(Objets)
        ## parametre servant à l'implémentation de la transmission
        Wd = np.zeros((len(X),len(X[0])))
        v_test = 0
        D_S = direction_souhaitee(n,D_S)

        ## force d'acceleration
        f_ac = (np.array(Allure)[:,None] * np.array(D_S) - V_ant)/ Tau
        f_obs = np.zeros((len(Objets),2))
        f_obj = np.zeros((len(Objets),2))

        ## force répulsive obstacle rectangulaire
        for j in range(len(Objets)):
            A = Dis_obstacle_rectangle(j)
            f_obs[j,0] = sum(A[:,0])
            f_obs[j,1] = sum(A[:,1])

        ## force répulsive objet
        for j in range(len(Objets)):
            A = Dis_objet_objet(j)
            f_obj[j,0] = sum(A[:,0])
            f_obj[j,1] = sum(A[:,1])
        if (affiche_temps):
            print(time.time() - t0, "force")
        t0 = time.time()

        ## prédiction de la vitesse
        f = f_ac + (1. / np.array(Masse)) * (f_obj + f_obs)
        for i in range(len(q)):
            if max(f[i]) > 10:
                D_S[i] = Direcs[rd.randint(0,7)]
        V_new = V_ant + h*f

        ## chocs
        qprime = q + (V_new+V_ant)*h/2
        for j in range(len(Objets)):
            q1 = q[j]
            q2 = qprime[j]
            rayon = Rayon[j]
            for obstacle in Obstacles_Rectangle:
                if ((q1[1] > obstacle[1] and q1[1] < obstacle[3]) and ( (q1[0] > obstacle[2] + rayon and q2[0] < obstacle[2] + rayon) or (q1[0] < obstacle[0] - rayon and q2[0] > obstacle[0] - rayon))):
                    V_new[j] = V_new[j] = [-1*V_new[j, 0], V_new[j, 1]]
                    D_S[j] = Direcs[rd.randint(0,7)]
                if ((q1[0] > obstacle[0] and q1[0] < obstacle[2]) and ( (q1[1] > obstacle[3] + rayon and q2[1] < obstacle[3] + rayon) or (q1[1] < obstacle[1] - rayon and q2[1] > obstacle[1] - rayon))):
                    V_new[j] = [V_new[j, 0], -1 * V_new[j, 1]]
                    D_S[j] = Direcs[rd.randint(0, 7)]

        ## correction de la vitesse
        for k in range(len(Objets)):
            z=zone(k)
            if z in list_st_dir:
                prob = rd.random()
                if prob < pd and p_s[k] == 0:
                    # Affichage
                    n_stand_infecte = n_stand_infecte + 1
                    n_infecte = n_infecte + 1
                    p_infecte.append(k)
                    p_s[k] = 1
            if z==-1 :
                q[k] = q[k] + (V_new + V_ant)[k] * h / 2
            elif StandNbr[z] < StandCap and Z[k][z] == Z0[k][z]:
                StandNbr[z] += 1
                Z[k][z] -= 1
            elif Z[k][z] == 1:
                    StandNbr[z] -= 1
                    Z[k][z] -= 1
            elif Z[k][z] < Z0[k][z] and Z[k][z] > 0:
                    Z[k][z] -= 1
            else :
                q[k] = q[k] + (V_new + V_ant)[k] * h / 2

        ## Evite les mouvements absurdes
        q_depasse = []
        for i in range(len(q)):
            if not in_cadre(q[i]):
                q_depasse.append(i)
        for p in q_depasse:
            x = q[p,0]
            y = q[p,1]
            if x < Xmin:
                q[p][0] = Xmin + 5
            if x > Xmax:
                q[p][0] = Xmax - 5
            if y < Ymin:
                q[p][1] = Ymin + 5
            if y > Ymax:
                q[p][ 1] = Ymax - 5
            if y > 60 and x > 60:
                if y>x :
                    q[p][0] = 55
                else:
                    q[p][1] = 55
        q_depasse = []
        for i in range(len(q)):
            if in_cadre(q[i]):
                q_depasse.append(q[i])
        if (affiche_temps):
            print(time.time() - t0, "correction vitesse")
        t0 = time.time()
        V_ant = V_new
        Dist = []
        for i in range(len(p_infectious)):
            list_dist = []
            for j in range(len(Objets)):
                r = p_infectious[i]
                if r != j :
                    list_dist.append(np.sqrt((q[r,0]-q[j,0])**2 + (q[r,1]- q[j,1])**2))
                else :
                    list_dist.append(10e99)
            Dist.append(list_dist)
        Dist = np.array(Dist)
        if (affiche_temps):
            print(time.time() - t0, "distance")
        t0 = time.time()

        ## implémentation de la transmission
        if len(p_infectious) != 0:
            for drr in range(len(p_infectious)):
                i = p_infectious[drr]
                ip = np.argmin([ abs(q[i,0] - x) for x in X[0]])
                jp = np.argmin([ abs(q[i,1] - y) for y in Y[:,0]])
                d = np.array(np.sqrt((X - X[jp, ip])**2 + (Y - Y[jp, ip])**2))
                MI = (d < d0)* 0.25 * (1 + np.cos(np.pi * d / d0))
                Wd = Wd + MI

        ## contamination directe
        if (n % int(1/h) == 0 ):
            if len(p_infectious) != 0 :
                if sum(p_s) != len(p_s) :
                    for p in range(len(p_infectious)):
                        mdist = Dist[p,:]
                        col = np.where(mdist < d_conta)[0]
                        col = list(set(col).difference(set(p_infecte)))
                        if len(col) != 0 :
                            prob = np.array([rd.random() for i in range(len(col))])
                            indf = np.where(prob < pd)[0]
                            if len(indf) != 0:
                                # Affichage
                                n_directe = n_directe + len(indf)
                                n_infecte = n_infecte + len(indf)
                                for p_inf in indf :
                                    if p_s[col[p_inf]] == 0:
                                        p_infecte.append(col[p_inf])
                                        p_s[col[p_inf]] = 1
        ## contamination indirecte
        if (n % int(1 / h) == 0):
            if len(p_infectious) != 0:
                if len(p_s) != 0:
                    for p in range(len(p_s)):
                        if p_s[p] == 0:
                            ip = np.argmin([abs(q[p, 0] - x) for x in X[0]])
                            jp = np.argmin([abs(q[p, 1] - y) for y in Y[:, 0]])
                            prob=rd.random()
                            p_a= 0.015 * C[jp, ip]
                            if p_a > prob :
                                # Affichage
                                v_test=v_test+1
                                n_indirecte=n_indirecte+1
                                n_infecte=n_infecte+1
                                p_s[p] = 1
                                p_infecte.append(p)
        C = C + h * (0.001 * Wd - sigma * C)
        nbr_infecte.append(n_infecte)
        nbr_indirecte.append(n_indirecte)
        nbr_directe.append(n_directe)
        nb_stand_infecte.append(n_stand_infecte)
        if (affiche_temps):
            print(time.time() - t0, "transmission")
    if affichage:
        plt.show()
    Instances_ss_distanciation.append([n_infecte,n_indirecte,n_directe,n_stand_infecte])
    fichier.write('Instance {0}\n'.format(iter))
    fichier.write( 'n_infecte = {0} '.format(n_infecte))
    fichier.write('n_indirecte = {0} '.format(n_indirecte))
    fichier.write('n_directe = {0} '.format(n_directe))
    fichier.write('n_stand_infecte = {0} '.format(n_stand_infecte))
    print([n_infecte,n_indirecte,n_directe,n_stand_infecte])
print(Instances_ss_distanciation)
fichier.close()