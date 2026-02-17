from scipy.stats import circmean,circvar
from cmath import phase
from  numpy import array
from scipy.stats import circmean,circvar,circstd
from numpy import *
from cmath import phase
from matplotlib.pylab import *

def decode(firing_rate, N_e):
    angles = np.arange(0,N_e)*2*np.pi/N_e
    R = []
    R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/N_e
    angle = np.angle(R)
    #if angle < 0:
    #    angle +=2*np.pi
    return angle, np.abs(R) 


def readout(i, t, sim_time, N_e):
    w1      = 100*ms
    w2      = 250*ms
    n_wins  = int((sim_time-w2)/w1)

    decs = []
    for ti in range(int(n_wins)):
        fr  = np.zeros(N_e)
        idx = ((t>ti*w1-w2/2) & (t<ti*w1+w2/2))
        ii  = i[idx]
        for n in range(N_e):
            fr[n] = sum(ii == n)
        dec = decode(fr, N_e)
        decs.append(dec)

    return decs, n_wins

def len2(x):
        if type(x) is not type([]):
                if type(x) is not type(np.array([])):
                        return -1
        return len(x)

def phase2(x):
        if not np.isnan(x):
                return phase(x)
        return np.nan

def circdist(angles1,angles2):
        if len2(angles2) < 0:
                if len2(angles1) > 0:
                        angles2 = [angles2]*len(angles1)
                else:
                        angles2 = [angles2]
                        angles1 = [angles1]             
        if len2(angles1) < 0:
                angles1 = [angles1]*len(angles2)
        return np.array(list(map(lambda a1,a2: phase2(np.exp(1j*a1)/np.exp(1j*a2)), angles1,angles2)))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def normgauss(xxx,sigma):
    gauss = (1/(sigma*np.sqrt(2*np.pi)) *np.exp(-(xxx-0)**2 / (2*sigma**2)))
    return gauss/gauss.max()


def normgrad(xxx):
    return np.gradient(xxx)/np.gradient(xxx).max()


def dog1(sigma,x):
    xxx     = np.arange(-200, 200, .1)
    dog_1st = normgrad(normgauss(xxx,sigma))
    return np.array(list(map(lambda x: dog_1st[find_nearest(xxx,x)], x)))


def rep_transform(prev,curr):
    global repstrength
    dist = np.degrees(circdist(np.radians(curr),np.radians(prev)))
    rep  = repstrength*dog1(45,dist) # or any other value that works
    print(curr, curr - rep)
    return curr - rep