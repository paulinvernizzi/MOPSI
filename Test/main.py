import numpy as np
import random as rd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display

#### INITIALISATION
if (True):
    Nb_indiv = 100
    pop = np.zeros((3, 16))
    pop[0] = np.array([0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00, 0.00, 0.00, 0.00]) / 100
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
    # Masse_m[2,:]=[1.22,0.488,1.16,5.192,5.192,5.152,5.152,4.728,4.728,3.782,3.782,3.70,3.70,4.492,4.492,4.314];
    # Masse_m[2,:]=[1.22,0.488,5.82,25.95,25.95,25.76,25.76,23.64,23.64,18.91,18.91,18.5,18.5,22.46,22.46,21.57]
    densite = 500
    ##Tableau contenant les rayons
    Rayon_m = np.sqrt(Masse_m / (500 * np.pi))
    ## Tableau contenant l'amplitude de la force de repulsion pour differentes classes d'age
    Taille = 16
    A_m = np.zeros((2, Taille))
    # A_m[1,:]=[10,10,75,110,122,127,142,142,180,225,247,247,290,305,305,305];
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
    ## Le nombre d'obstacles & Sorties
    Obstacles_Rectangles = []
    ##Les stands
    Stands=[]
    ## la durï¿½e de simulation, le pas de temps espace et le nombre d'itï¿½rations
    T = 300
    h = 1e-1;  # pas du temps
    N_iter = T / h
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
    ## Les parametres de l'equation de contact
    K_N_obj = 1000
    K_T_obj = 0
    K_T_obs = 0
    K_N_obs = 1e5
    K_T = 0
    K_N = 0
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
    d_obs = 1.8
    R_obs = 10
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
    Obstacles_Rectangle = []
    Stand=[]
    Zone_Interdite = []
    # Duree de visite du centre commercial
    T_d = []
    ## Matrices pour la transmission du virus
    p_infecte = []
    p_infectious = []
    p_s = []
    nbr_s = []
    nbr_infecte = []
    nbr_indirecte = []
    nbr_directe = []
    ## Parmetres des equations de Concentration du virus
    sigma = (0.1732 / 3600);  # duree de vie du virus sur les surfaces
    d0 = 2;  # distance maximale a  laquelle une surface est contaminee
    pd = 0.04
    hx = 0.5;  # le pas de discretisation

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
    # Initialisation de la matrice contenant les indices des individus
    # infectieux
    p_infectious = []
    t = 0
    ##matrice contenant le nombre de contamination pour chaque classe d'age
    Matrix = np.zeros((1, 16))

#### FONCTIONS ANNEXES
def direction_souhaitee(n,D_S):
    Nb_objets = len(Objets)
    if (n == 1):
        choice = np.random.randint(1, 8, size = Nb_objets)
        D_S = np.array([Direcs[choix] for choix in choice])
    if (n % 10 == 0):
        aa = np.random.randint(1, Nb_objets, size = int(np.floor(Nb_objets * 0.8)))
        choice = np.random.randint(1, 8, size = int(np.floor(Nb_objets * 0.8)))
        D_S[aa] = Direcs[choice]
    return D_S

def Dis_obstacle_rectangle(j):
    A = []
    Objet = Objets[j]
    rayon = Rayon[j]
    for obstacle in Obstacles_Rectangle:
        x1,y1,x2,y2 = obstacle
        dy = ( (y2 - Objet[1])*(y2 > Objet[1]) + (y1 - Objet[1])*(y1 < Objet[1]) ) * (Objet[0] > x1 and Objet[0] < x2) + 10e99 * (Objet[0] <= x1 or Objet[0] >= x2)
        dx = ( (Objet[0] - x2)*(Objet[0] > x2) + (x1 - Objet[0])*(Objet[0] < x1) ) * (Objet[1] < y2 and Objet[1] > y1) + 10e99 * (Objet[1] <= y1 or Objet[1] >= y2)
        d = np.sqrt(dx**2 + dy **2)
        A.append([(d_obs > dx)*A_obs*np.exp((d_obs - d)/B_obs)*(-1)*(dx <= 0),(d_obs > dy)*A_obs*np.exp((d_obs - d)/B_obs)*(-1)*(dy<=0) ])
        return np.array(A)

def Dis_objet_objet(i):
    A = []
    Objet = Objets[i]
    rayon = Rayon[i]
    dir = D_S[i]
    for j in range(len(Objets)):
        x2,y2 = Objet
        x1,y1 = Objets[j]
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
    global Obstacles_Rectangles
    Obstacles_Rectangle.append([x,y,x+longueur,y+largeur])

def Ajouter_Zone_Interdite(x,y,largeur,longueur):
    Zone_Interdite.append([x,y,x+largeur,y+longueur])

def Ajouter_Objet(x,y,ray,cod,allure):
    global Objets
    global Rayon
    global Masse
    global Code
    global Allure
    global d_obj
    global A_obj
    global B_obj
    global Tau
    global T_d

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
    T_d.append(T_m[0,cod] + T_m[1,cod]*rd.gauss(0,1))

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
    Ajouter_Obstacle_Rectangle(60,0,50,10)
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
    #obstacke représentant l'entrée
    #obstacle représentant la salle à droite
    Ajouter_Obstacle_Rectangle(76,Ymin+5,eps,25)
    Ajouter_Obstacle_Rectangle(76,40,eps,20)
    #ajout des personnes
    for i in range(len(pop[2])):
        Ajouter_Plusieurs_Objets(int(pop[1,i]),i)
    q = np.array(Objets)
    p_infectious = rd.sample([i for i in range(len(Objets))],3)
    ### Initialisation de la matrice contenant les indices des susceptibles
    p_s = np.array(range(len(Objets)))
    p_s = p_s.copy()
    p_s = list(set(p_s).difference(set(p_infectious)))
    D_S = np.array(range(len(Objets)),dtype= tuple)

V_ant = np.zeros((len(Objets),2))
ims = []
fig, ax = plt.subplots()
while (n < N_iter + 1 and len(Objets) > 0):
    ax.cla()
    ax.scatter(0, 0, c='w')
    for j in range(len(q)):
        ax.add_patch(plt.Circle((q[j, 0], q[j, 1]), Rayon[j], color="purple"))
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
        ax.add_patch(patches.Rectangle((stand[0],stand[1]),0.2,stand[3],color="white"))
        ax.add_patch(patches.Rectangle((stand[0], stand[1]), stand[2], 0.2, color="white"))
        ax.add_patch(patches.Rectangle((stand[0]+stand[2], stand[1]), 0.2, stand[3], color="white"))
        ax.add_patch(patches.Rectangle((stand[0], stand[1]+stand[3]), stand[2], 0.2, color="white"))
    ax.set(xlim = (Xminn,Xmaxn),ylim = (Yminn, Ymaxn))
    im = ax.imshow(C,origin = "lower",cmap = plt.cm.rainbow, extent = (Xminn,Xmaxn,Yminn,Ymaxn),animated = True)
    ims.append([im])
    plt.pause(0.00001)
    n = n + 1
    t = t + h
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

    ## force attractive stands 
  

    if (n == 1):
        continue
    ## prédiction de la vitesse
    f = f_ac + 1. / np.array(Masse) * (f_obj + f_obs)
    V_new = V_ant + h * f

    ## chocs
    qprime = q + (V_new+V_ant)*h/2
    for j in range(len(Objets)):
        q1 = Objets[j]
        q2 = qprime[j]
        rayon = Rayon[j]
        for obstacle in Obstacles_Rectangle:
            if ((q1[1] > obstacle[1] and q1[1] < obstacle[3]) and ( (q1[0] > obstacle[2] + rayon and q2[0] < obstacle[2] + rayon) or (q1[0] < obstacle[0] - rayon and q2[0] > obstacle[0] - rayon))):
                V_new[j] = V_new[j] = [-1*V_new[j, 0], V_new[j, 1]]
            if ((q1[0] > obstacle[0] and q1[0] < obstacle[2]) and ( (q1[1] > obstacle[3] + rayon and q2[1] < obstacle[3] + rayon) or (q1[1] < obstacle[1] - rayon and q2[1] > obstacle[1] - rayon))):
                V_new[j] = [V_new[j, 0], -1 * V_new[j, 1]]

    ## correction de la vitesse
    q = q + (V_new+V_ant)*h/2
    V_ant = V_new
    Dist = []
    for i in range(len(Objets)):
        list_dist = []
        for j in range(len(Objets)):
            if i != j :
                list_dist.append(np.sqrt((q[i,0]-q[j,0])**2 + (q[i,1]- q[j,1])**2))
            else :
                list_dist.append(10e99)
        Dist.append(list_dist)
    Dist = np.array(Dist)

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
    if len(p_infectious) != 0 :
        if len(p_s) != 0 :
            for p in p_infectious:
                mdist = Dist[p,:]
                col = np.where(mdist<5)[0]
                col = list(set(col).difference(set(p_infecte)))
                if len(col) != 0 :
                    prob = np.array([rd.random() for i in range(len(col))])
                    indf = np.where(prob < pd)[0]
                    if len(indf) != 0:
                        print("ind infecté !")
                        # Affichage
                        v_test = v_test + 1
                        n_directe = n_directe + len(indf)
                        n_infecte = n_infecte + len(indf)
                        #nbr_directe = [nbr_directe, t, n_directe];
                        #nbr_infecte = [nbr_infecte, t, n_infecte];
                        #p_infecte = [p_infecte, col(indf)]
                        #co = Code(col(indf))
                        #categ = categorical(co, [1:16], {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'})
                        #Matrix = Matrix + histcounts(categ)
                        for p_inf in indf :
                            p_infecte.append(col[p_inf])

    ## contamination indirecte
    if len(p_infectious) != 0:
        if len(p_s) != 0:
            for p in p_s:
                ip = np.argmin([abs(q[i, 0] - x) for x in X[0]])
                jp = np.argmin([abs(q[i, 1] - y) for y in Y[:, 0]])
                prob=rd.random()
                p_a= 0.015 * C[jp, ip]
                if p_a > prob :
                    print("ind infecté")
                    # Affichage
                    v_test=v_test+1
                    n_indirecte=n_indirecte+1
                    #nbr_indirecte=[nbr_indirecte;t, n_indirecte];
                    n_infecte=n_infecte+1
                    #nbr_infecte=[nbr_infecte;t, n_infecte];
                    #p_infecte=[p_infecte;p_s(indf)];
                    #co=Code(p_s(indf));
                    #categ = categorical(co, [1:16], {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'})
                    #Matrix = Matrix + histcounts(categ)
                    p_s.pop(p)
                    p_infecte.append(p)
    #if v_test == 0 :
        #nbr_directe = [nbr_directe;t, n_directe];
        #nbr_indirecte = [nbr_indirecte;t, n_indirecte];
        #nbr_infecte = [nbr_infecte;t, n_infecte];
    C = C + h * (0.001 * Wd - sigma * C)
plt.show()