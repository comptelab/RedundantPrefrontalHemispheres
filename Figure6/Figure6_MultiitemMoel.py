"""
Implementation of a working memory model.
Literature:
Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and
network dynamics underlying spatial working memory in a cortical network model.
Cerebral Cortex, 10(9), 910-923.

Some parts of this implementation are inspired by material from
*Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
online available.

Note: Most parameters differ from the original publication.
"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

''' DISCLAIMER: this script generates 1 simulation of the model. For multiple generation use:
    >> parallel python wm_model_STDP.py ::: `seq 1 num_trials` << at best in a cluster'''

from brian2 import ms, Hz, namp, nF, nS, mV, kHz, second
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation, defaultclock, run, prefs, seed, start_scope, clear_cache
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from scipy.special import erf
from numpy.fft import rfft, irfft
import os
import sys
import time
import socket
import random
from joblib import Parallel, delayed
import multiprocessing as mp
from model_fcts import *
from random import randint

import seaborn as sns
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib.pylab as pylab

#clear_cache('cython')

import seaborn as sns
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib.pylab as pylab

from brian2 import ms, Hz, namp, nF, nS, mV, kHz, second
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation, defaultclock, run, prefs, seed, start_scope, clear_cache
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from scipy.special import erf
from numpy.fft import rfft, irfft
import os
import sys
import time
import socket
import random
from joblib import Parallel, delayed
import multiprocessing as mp
from model_fcts import *
from random import randint

import seaborn as sns
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib.pylab as pylab

#clear_cache('cython')

import seaborn as sns
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'
import matplotlib.pylab as pylab

import matplotlib.pyplot as plt
plt.style.use("stylefile.mplstyle")


# Define colors
colors = {
    "Single": "#333333",
    "Left": "#246EB9",
    "Right": "#8B1E3F",
    "Ipsilateral": "#1B9E77",
    "Contralateral": "#D95F02",
    "Within": "#2B4162",
    "Across": "#E89D0B"
}


prefs.codegen.target = 'cython'


filename     = "TwoAreaMultiitem_002_ipsi80"

start_time = time.time()
numcores   = np.max([int(mp.cpu_count()/2)-2, 1])
if numcores <8:
    numcores=1

# in range [0,1] for 0-connected, 0.5 as much connected as within a hemisphere, 1 only across
#across_factor = 0.003#0.05#0

# STP PARAMETERS
U_0   = 0.15#0.2#0.2                            # Synaptic release probability at rest
tau_d = 250*ms#4000*ms#2500 * ms#200 * ms      # Synaptic depression time constant
tau_f = 1500*ms#5000*ms#4000*ms#1500 * ms       # Synaptic facilitation time constant

time_par = 500                                 # length of reignition current #300

reig_strength = 0.036*namp#0.039 * namp  #0.042 * namp                     # non-specific read-out current # 0.04 * namp


repstrength = 1.25

defaultclock.dt = 0.04 * ms

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
    xxx     = np.arange(-2*np.pi, 2*np.pi, .0001)
    dog_1st = normgrad(normgauss(xxx,sigma))
    return np.array(list(map(lambda x: dog_1st[find_nearest(xxx,x)], x)))

####################################################################################
#                                   START SIMULATIONS                              #
####################################################################################

def run_simulation(across_factor):
    start_scope()
    global rep, U_0, tau_d, tau_f, time_par, reig_strength

    across_factor = np.round(across_factor, 3)
    #print(across_factor)

    N_e                 = 1024
    N_i                 = 256

    ds=1000*ms
    dl=3000*ms

    t_delay1_duration   = dl#(int(round(rand()))*(dl-ds)+ds)
    t_delay2_duration   = dl#(int(round(rand()))*(dl-ds)+ds)

    defaultclock.dt     = 0.1 * ms
    t_stimulus1_start   = 1000*ms
    t_stimulus_duration = 300*ms#500*ms#125 * ms
    t_stimulus1_end     = t_stimulus1_start + t_stimulus_duration
    t_delay1_end        = t_stimulus1_end+t_delay1_duration#t_distr_end + t_delay1_duration
    t_response_start    = t_delay1_end
    t_response_end      = t_response_start#t_stimulus_duration

    t_fix_duration      = 1500*ms

    t_reig_duration     = time_par * ms
    t_reig_start        = t_response_end+ t_fix_duration#
    t_reig_end          = t_reig_start+t_reig_duration

    t_stimulus2_start   = t_reig_end + 1200*ms
    t_stimulus2_end     = t_stimulus2_start + t_stimulus_duration
    t_delay2_end        = t_stimulus2_end+t_delay2_duration
    #t_reig2_start       = t_delay2_end
    #t_reig2_end		    = t_reig2_start+t_reig_duration
    t_response2_end     = t_delay2_end

    # define length of simulation
    sim_time            = t_response_end+500*ms#t_response2_end#t_delay2_end#t_stimulus2_end+500*ms

    # decide if trial is sub or supraliminal
    trial1 = 1#randint(0,1)
    trial2 = 1#randint(0,1)

    REACT1 = True#np.random.choice([True, False])
    REACT2 = True#np.random.choice([True, False])

    #print(across_factor)
    # for simulation in range(simulations):
    #     print(simulation)
    #     time.sleep(0.1)
    #print('start')

    # resets brians internal seed so that the simulations are not always the same
    seed()
    simulate_wm(N_excitatory=N_e, N_inhibitory=N_i,
              stimulus_strength=0.3* namp, Jpos_excit2excit=1.73,#1.77,1.83 #1.705 in non-connected
              across_factor = across_factor,
              #stimulus1_center_deg=stim1A_location, stimulus2_center_deg=stim2_location,
              #distractor_center_deg=distr_location,
              t1=trial1, t2=trial2,
              t_stimulus1_start=t_stimulus1_start, t_stimulus2_start=t_stimulus2_start,
              t_stimulus_duration=t_stimulus_duration,
              t_delay1=t_delay1_duration, t_delay2=t_delay2_duration,
              t_reig_start=t_reig_start, t_reig_end=t_reig_end,
              REACT1 = REACT1, REACT2= REACT2,
              t_response_end = t_response_end, t_response2_end = t_response2_end,
              t_iti_duration=t_fix_duration,
              sim_time=sim_time)

    return


################################################################################
#                              DEFINE SIMULATIONS                              #
################################################################################


def simulate_wm(
        N_excitatory=2048, N_inhibitory=512,
        N_extern_poisson=1000, poisson_firing_rate=0.925*Hz,#1.15 * Hz,
        sigma_weight_profile=14.4, Jpos_excit2excit=1.64, across_factor=0.03,
        #stimulus1_center_deg=180, stimulus2_center_deg=235,
        #distractor_center_deg = 90,
        stimulus_width_deg=60, stimulus_strength=0.04 * namp,
        t1=1, t2=1,
        t_stimulus1_start=0 * ms, t_stimulus2_start=0 * ms,
        t_stimulus_duration=0 * ms,
        t_delay1=0 * ms, t_delay2=0 * ms,
        t_reig_start=0 * ms, t_reig_end=0 * ms,
        REACT1=True, REACT2=True,
        t_reig2_start=0 * ms, t_reig2_end=0 * ms,
        t_response_end=0*ms, t_response2_end = 0*ms,
        t_iti_duration=0 * ms, sim_time=0 * ms,
        monitored_subset_size=1024):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """

    global time_par, reig, U_0, tau_d, tau_f

    # specify the excitatory pyramidal cells:
    Cm_excit                    = 0.5 * nF  # membrane capacitance of excitatory neurons
    G_leak_excit                = 25.0 * nS  # leak conductance
    E_leak_excit                = -70.0 * mV  # reversal potential
    v_firing_threshold_excit    = -50.0 * mV  # spike condition
    v_reset_excit               = -60.0 * mV  # reset voltage after spike
    t_abs_refract_excit         = 2.0 * ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib                    = 0.2 * nF
    G_leak_inhib                = 20.0 * nS
    E_leak_inhib                = -70.0 * mV
    v_firing_threshold_inhib    = -50.0 * mV
    v_reset_inhib               = -60.0 * mV
    t_abs_refract_inhib         = 1.0 * ms

    # specify the AMPA synapses
    E_AMPA      = 0.0 * mV
    tau_AMPA    = 2.0 * ms

    # specify the GABA synapses
    E_GABA      = -70.0 * mV
    tau_GABA    = 10.0 * ms

    # specify the NMDA synapses
    E_NMDA      = 0.0 * mV
    tau_NMDA_s  = 100.0 * ms
    tau_NMDA_x  = 2.0 * ms
    alpha_NMDA  = 0.5 * kHz

    weight_scaling_factor = 2048./N_excitatory

    # projections from the external population
    G_extern2inhib  = 2.38 * nS
    G_extern2excit  = 3.1 * nS

    # precompute the weight profile for the recurrent population
    tmp = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel = [(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *
                             math.exp(-.5 * (360. * min(nj,
                                                        N_excitatory - nj) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
                            for nj in
                            range(N_excitatory)]
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)

    # projectsions from the inhibitory populations
    G_inhib2inhib   = weight_scaling_factor * 1.024 * nS
    G_inhib2excit   = weight_scaling_factor * 1.336 * nS

    # projections from the excitatory population NMDA
    G_excit2excit   = 1.17*weight_scaling_factor * 1.05 * 0.28 * nS #nmda+ampa #
    G_excit2inhib   = weight_scaling_factor * 0.212 * nS  # nmda+ampa

    # recurrent AMPA
    G_excit2excitA  = weight_scaling_factor * 2.7 * 0.251 * nS  #ampa
    #GEEA            = G_excit2excitA/G_extern2excit
    G_excit2inhibA  = weight_scaling_factor * 0.192 * nS  #ampa
    #GEIA            = G_excit2inhibA/G_extern2inhib

    # across areas
    G_excit2excitA_across = G_excit2excitA * across_factor
    GEEA_across = G_excit2excitA_across / G_extern2excit
    G_excit2inhibA_across = G_excit2inhibA * across_factor
    GEIA_across     = G_excit2inhibA_across/G_extern2inhib

    # reduce within connections by same amount as across is increased to stay in similar parameter range
    if np.sign(GEEA_across) == 1: # if tuned, subtract max
        G_excit2excitA = G_excit2excitA - np.abs(G_excit2excitA_across)
        G_excit2inhibA = G_excit2inhibA - np.abs(G_excit2inhibA_across)
        reduceG_untuned = 0# don't subtract anything from later computed weight kernel
    else:
        reduceG_untuned = np.mean(np.abs(GEEA_across) *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *\
                                                          exp(-.5 * (360. * abs(np.linspace(-512,512, 1025)) /\
                                                                     N_excitatory) ** 2 / sigma_weight_profile ** 2)))

    GEEA            = G_excit2excitA/G_extern2excit
    GEIA            = G_excit2inhibA/G_extern2inhib


    # define the inhibitory population
    a = 0.062/mV
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-a*v)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v     = np.random.uniform(v_reset_inhib / mV, high=v_firing_threshold_inhib / mV,
                                       size=N_inhibitory) * mV

    # set the connections: inhib2inhib
    syn_inhibinhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * ms, method="rk2")
    syn_inhibinhib.connect(condition="i!=j", p=1.0)
#    syn_inhib2inhib.connect(p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    #####################################
    #        Inhib AREA 2               #
    #####################################

    inhib_pop2 = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop2.v = np.random.uniform(v_reset_inhib / mV, high=v_firing_threshold_inhib / mV,
                                    size=N_inhibitory) * mV

    # set the connections: inhib2inhib
    syn_inhib2inhib2 = Synapses(inhib_pop2, target=inhib_pop2, on_pre="s_GABA += 1.0", delay=0.0 * ms, method="rk2")
    syn_inhib2inhib2.connect(condition="i!=j", p=1.0)
    #    syn_inhib2inhib.connect(p=1.0)
    # set the external input
    input_ext2inhib2 = PoissonInput(target=inhib_pop2, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    ######################################
    #          EXCITATION                #
    ######################################

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-a*v)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit, method="rk2")

    # initialize with random voltages:
    excit_pop.v = np.random.uniform(v_reset_excit / mV, high=v_firing_threshold_excit / mV,
                                       size=N_excitatory) * mV
    excit_pop.I_stim = 0. * namp
    excit_pop.s_NMDA = 0.05
    excit_pop.x=0.0017
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhibexcit = Synapses(inhib_pop, excit_pop, on_pre="s_GABA += 1.0", method="rk2")
    syn_inhibexcit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excitinhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excitinhib.connect(p=1.0)

    ###################################
    #           AREA 2                #
    ###################################
    excit_pop2 = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop2.v = np.random.uniform(v_reset_excit / mV, high=v_firing_threshold_excit / mV,
                                    size=N_excitatory) * mV
    excit_pop2.I_stim = 0. * namp
    excit_pop2.s_NMDA = 0.05
    excit_pop2.x = 0.0017
    # set the connections: extern2excit
    input_ext2excit2 = PoissonInput(target=excit_pop2, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit2 = Synapses(inhib_pop2, excit_pop2, on_pre="s_GABA += 1.0", method="rk2")
    syn_inhib2excit2.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib2 = Synapses(excit_pop2, inhib_pop2,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib2.connect(p=1.0)

    # TODO! make this (clock-driven) if you care about the exact decay of the synaptic traces!! (But takes a lot more time)
    synapses_eqs = '''
    w : 1
    # Usage of releasable neurotransmitter per single action potential:
    du_S/dt =  (U_0 -u_S)/tau_f     : 1 (event-driven)
    # Fraction of synaptic neurotransmitter resources available:
    dx_S/dt = (1.0 - x_S)/tau_d : 1 (event-driven)
    '''
    synapses_action = '''
    r_S = u_S * x_S 
    x_S -= r_S
    u_S += U_0 * (1.0 - u_S)
    '''
    syn_excitexcit = Synapses(excit_pop, excit_pop, model=synapses_eqs,
                       on_pre=synapses_action+'s_AMPA_post += w*r_S; x_pre += (1.0/N_excitatory)', method="rk2")
    syn_excitexcit.connect(condition='i!=j', p=1.0)
    syn_excitexcit.w['abs(i-j)<N_excitatory//2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile ** 2)) - reduceG_untuned'
    syn_excitexcit.w['abs(i-j)>=N_excitatory//2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile ** 2)) - reduceG_untuned'

    # Start from "resting" condition: all synapses have fully-replenished
    # neurotransmitter resources
    u_start = 0.08#0.16#U_0#0.21#
    x_start = 0.9#0.99#1.0#0.95#

    syn_excitexcit.u_S = u_start
    syn_excitexcit.x_S = x_start

    syn_excitinhibA = Synapses(excit_pop, inhib_pop, model="w : 1", on_pre="s_AMPA_post += w", method="rk2")
    syn_excitinhibA.connect(p=1.0)
    syn_excitinhibA.w = GEIA

    #########################################
    #                AREA 2                 #
    #########################################
    syn_excit2excit2 = Synapses(excit_pop2, excit_pop2, model=synapses_eqs,
                              on_pre=synapses_action + 's_AMPA_post += w*r_S; x_pre += (1.0/N_excitatory)', method="rk2")
    syn_excit2excit2.connect(condition='i!=j', p=1.0)
    syn_excit2excit2.w[
        'abs(i-j)<N_excitatory//2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile ** 2)) - reduceG_untuned'
    syn_excit2excit2.w[
        'abs(i-j)>=N_excitatory//2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile ** 2)) - reduceG_untuned'
    # Start from "resting" condition: all synapses have fully-replenished
    # neurotransmitter resources
    syn_excit2excit2.u_S = u_start
    syn_excit2excit2.x_S = x_start

    syn_excit2inhib2A = Synapses(excit_pop2, inhib_pop2, model="w : 1", on_pre="s_AMPA_post += w", method="rk2")
    syn_excit2inhib2A.connect(p=1.0)
    syn_excit2inhib2A.w = GEIA

    ##################################################################
    #                           CONNECT AREAS                        #
    ##################################################################
    # Area 1 to Area 2:

    # excitatory
    syn_excitexcit2 = Synapses(excit_pop, excit_pop2,
                               model="w:1", on_pre='s_AMPA_post += w', method="rk2")
    syn_excitexcit2.connect(condition='i!=j', p=1.0)
    if np.sign(GEEA_across) == 1: # if positive, tuned connected
        syn_excitexcit2.w[
            'abs(i-j)<N_excitatory//2'] = 'GEEA_across *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'
        syn_excitexcit2.w[
            'abs(i-j)>=N_excitatory//2'] = 'GEEA_across *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'
    else: # GEEA_across negative means untuned connected, connect with mean strength of tuned connections
        syn_excitexcit2.w = np.mean(np.abs(GEEA_across) *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *\
                                                          exp(-.5 * (360. * abs(np.linspace(-512,512, 1025)) /\
                                                                     N_excitatory) ** 2 / sigma_weight_profile ** 2)))
    # inhibitory
    syn_excitinhib2 = Synapses(excit_pop, inhib_pop2,
                               model="w:1", on_pre='s_AMPA_post += w', method="rk2")
    syn_excitinhib2.connect(condition='i!=j', p=1.0)
    syn_excitinhib2.w = GEIA_across


    # Area 2 to Area 1:

    # excitatory
    syn_excit2excit = Synapses(excit_pop2, excit_pop,
                               model="w:1", on_pre='s_AMPA_post += w', method="rk2")
    syn_excit2excit.connect(condition='i!=j', p=1.0)
    if np.sign(GEEA_across) == 1:  # if positive, tuned connected
        syn_excit2excit.w[
            'abs(i-j)<N_excitatory//2'] = 'GEEA_across *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'
        syn_excit2excit.w[
            'abs(i-j)>=N_excitatory//2'] = 'GEEA_across *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'
    else:  # GEEA_across negative means untuned connected, connect with mean strength of tuned connections
        syn_excitexcit2.w = np.mean(np.abs(GEEA_across) * (Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * \
                                                           exp(-.5 * (360. * abs(np.linspace(-512, 512, 1025)) / \
                                                                      N_excitatory) ** 2 / sigma_weight_profile ** 2)))
    # inhibitory
    syn_excit2inhib = Synapses(excit_pop2, inhib_pop,
                               model="w:1", on_pre='s_AMPA_post += w', method="rk2")
    syn_excit2inhib.connect(condition='i!=j', p=1.0)
    syn_excit2inhib.w = GEIA_across

    # set the STRUCTURED recurrent NMDA input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = np.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total,N_excitatory)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

        #area 2
        fft_s_NMDA2 = rfft(excit_pop2.s_NMDA)
        fft_s_NMDA_total2 = np.multiply(fft_presyn_weight_kernel, fft_s_NMDA2)
        s_NMDA_tot2 = irfft(fft_s_NMDA_total2, N_excitatory)
        excit_pop2.s_NMDA_total_ = s_NMDA_tot2

        # excit_pop.s_NMDA_total = s_NMDA_tot
        # inhib_pop.s_NMDA_total = fft_s_NMDA[0]

    # @network_operation(dt=100 * ms)
    # def time_counter(t):
    #     print(t)

    @network_operation(dt=1 * ms)
    def stimulate_network(t):
        if t >= t_stimulus1_start and t < t_stimulus1_start + t_stimulus_duration:  # Stimulus1
            xxx = np.linspace(-np.pi, np.pi, len(stim1A_target_idx))
            ipsi_percentage = 0.8
            # stimulus 1A
            if HEMI_A == 1:
                # contralateral -> excit_pop (left) gets fully excited by HEMI==1 (right)
                excit_pop.I_stim[stim1A_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2)
                # ipsilateral -> excit_pop2 (right) gets fully excited by HEMI==0 (left) but not by HEMI==1 (right)
                excit_pop2.I_stim[stim1A_target_idx] = ipsi_percentage * stimulus_strength * normgauss(xxx, np.pi / 2)
            else:
                excit_pop2.I_stim[stim1A_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2)
                excit_pop.I_stim[stim1A_target_idx] = ipsi_percentage * stimulus_strength * normgauss(xxx, np.pi / 2)

            # #stimulus 1B
            if HEMI_B == 1:
                # contra
                excit_pop.I_stim[stim1B_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2)
                # ipsi
                excit_pop2.I_stim[stim1B_target_idx] = ipsi_percentage * stimulus_strength * normgauss(xxx, np.pi / 2)
            else:
                excit_pop2.I_stim[stim1B_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2) # contra
                excit_pop.I_stim[stim1B_target_idx] = ipsi_percentage * stimulus_strength * normgauss(xxx, np.pi / 2) # ipsi
        elif t >= t_response_end and t < t_response_end + 1 * t_stimulus_duration:  # Off signal btwn trials
            excit_pop.I_stim = -2. * stimulus_strength  # -1.*stimulus_strength
            excit_pop2.I_stim = -2. * stimulus_strength  # -1.*stimulus_strength
        elif t >= t_reig_start and t < t_reig_end and REACT1==True and REACT2==True :  # Reactivation
            excit_pop.I_stim = reig_strength
            excit_pop2.I_stim = reig_strength
        elif t >= t_reig_start and t < t_reig_end and REACT1==True:  # Reactivation
            excit_pop.I_stim = reig_strength
        elif t >= t_reig_start and t < t_reig_end and REACT2==True:  # Reactivation
            excit_pop2.I_stim = reig_strength
        elif t > (t_reig_end+300* ms) and t < (t_reig_end+800* ms):                                      # Reactivation
           excit_pop.I_stim = -1. * stimulus_strength
           excit_pop2.I_stim = -1. * stimulus_strength
        elif t >= t_stimulus2_start and t < t_stimulus2_start + t_stimulus_duration:  # Stimulus2
            xxx = np.linspace(-np.pi, np.pi, len(stim2_target_idx))
            excit_pop.I_stim[stim2_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2)
            excit_pop2.I_stim[stim2_target_idx] = stimulus_strength * normgauss(xxx, np.pi / 2)
        elif t >= t_reig2_start and t < t_reig2_end and t2 == 0:  # Reignition current
            excit_pop.I_stim = reig_strength
            excit_pop2.I_stim = reig_strength
        else:
            excit_pop.I_stim = 0. * namp
            excit_pop2.I_stim = 0. * namp

    def get_monitors(pop, syn, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = [int(math.ceil(k))
             for k in np.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        spike_monitor   = SpikeMonitor(pop, record=idx_monitored_neurons)
        synapse_monitor = StateMonitor(syn, "u_S", dt=10*ms, record=syn[:,0])#StateMonitor(syn_excit2excit, "stp", record=syn_excit2excit[stim1A_center_idx,stim1_center_idx-10:stim1_center_idx+10], dt=1*ms)
        depression_monitor = StateMonitor(syn, "x_S", dt=10*ms, record=syn[:,0])

        return spike_monitor,synapse_monitor, depression_monitor

    # collect data of a subset of neurons:
    spike_monitor_excit, synapse_monitor, depression_monitor_excit = get_monitors(excit_pop, syn_excitexcit,\
                                                                                  monitored_subset_size, N_excitatory)
    spike_monitor_excit2, synapse_monitor2, depression_monitor_excit2 = get_monitors(excit_pop2, syn_excit2excit2, \
                                                                                  monitored_subset_size, N_excitatory)

    ######################################################################################
    #                                     RUN SIMULATIONS                                #
    ######################################################################################

    stim1A_location      = 45#45#np.random.choice(stim_choices)#randint(0,359)#180
    stim1B_choices = [110]#[110, -110]
    stim1B_location      = (stim1A_location+np.random.choice(stim1B_choices))%360#
    distr_location      = 0#randint(0,359)
    stim2_location_orig      = randint(0,359)#135

    # define hemifields
    HEMI_A, HEMI_B = 0, 0
    if (stim1A_location < 90) | (stim1A_location > 270):
        HEMI_A = 1
    if (stim1B_location < 90) | (stim1B_location > 270):
        HEMI_B = 1

    sigma = 0.5
    max_repulsion = 0.75
    # add a manual adaptation of DoG shape (i.e. perturbation from visual areas)
    stim2_location = stim2_location_orig + max_repulsion * dog1(sigma, [np.deg2rad(stim1A_location-stim2_location_orig)])[0]

    # compute the simulus index
    stim1A_center_idx = int(np.round(N_excitatory / 360. * stim1A_location))
    stim1_width_idx  = int(np.round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim1A_target_idx = [idx % N_excitatory
                       for idx in
                       range(stim1A_center_idx - stim1_width_idx, stim1A_center_idx + stim1_width_idx + 1)]

    # compute the simulus index
    stim1B_center_idx = int(np.round(N_excitatory / 360. * stim1B_location))
    stim1B_target_idx = [idx % N_excitatory
                        for idx in
                        range(stim1B_center_idx - stim1_width_idx, stim1B_center_idx + stim1_width_idx + 1)]


    stim2_center_idx = int(np.round(N_excitatory / 360. * stim2_location))
    stim2_width_idx  = int(np.round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim2_target_idx = [idx % N_excitatory
                       for idx in
                       range(stim2_center_idx - stim2_width_idx, stim2_center_idx + stim2_width_idx + 1)]

    run(sim_time)
    evaluate_wm(across_factor, spike_monitor_excit, synapse_monitor, depression_monitor_excit, \
                spike_monitor_excit2, synapse_monitor2, depression_monitor_excit2, \
                N_excitatory, stimulus_strength, t1, t2, stim1A_location, stim1B_location,\
                HEMI_A, HEMI_B, distr_location, stim2_location_orig,\
                    t_stimulus1_start, t_stimulus2_start, t_stimulus_duration, t_delay1,\
                      t_delay2, t_reig_start, t_reig_end, t_reig2_start, t_reig2_end, REACT1, REACT2, t_response_end,\
                          t_response2_end, t_iti_duration, sim_time)

    return


####################################################################################
#                                EVALUATE SIMULATIONS                              #
####################################################################################

def evaluate_wm(across_factor, spike_monitor_excit,synapse_monitor_excit, depression_monitor_excit,\
                spike_monitor_excit2,synapse_monitor_excit2, depression_monitor_excit2, \
                N_e, stimulus_strength, trial1, trial2, stim1A_location, stim1B_location,\
                HEMI_A, HEMI_B,\
                distr_location, stim2_location, t_stimulus1_start,\
                    t_stimulus2_start, t_stimulus_duration, t_delay1_duration, t_delay2_duration,\
                            t_reig_start, t_reig_end, t_reig2_start, t_reig2_end,  REACT1, REACT2,\
                t_response_end, t_response2_end, t_iti_duration, sim_time):



    ds=1000*ms
    dl=3000*ms

    t_stimulus1_end = t_stimulus1_start+t_stimulus_duration
    t_stimulus2_end = t_stimulus2_start+t_stimulus_duration

    i,t         = spike_monitor_excit.it
    i2,t2         = spike_monitor_excit2.it
    depr        = depression_monitor_excit.x_S
    depr2       = depression_monitor_excit2.x_S
    facil         = synapse_monitor_excit.u_S
    facil2 = synapse_monitor_excit2.u_S
    stp_val = np.multiply(facil, depr)
    stp_val2 = np.multiply(facil2, depr2)

    n_stim=np.zeros(N_e)
    n=np.zeros(N_e)
    n_react = np.zeros(N_e)
    n2=np.zeros(N_e)
    n_area2 = np.zeros(N_e)
    n_react_area2 = np.zeros(N_e)
    n2_area2 = np.zeros(N_e)
    for k in range(0,N_e):
        # area1
        n_stim[k] = np.where(i[np.where(np.logical_and(t>=t_stimulus1_start, t<t_stimulus1_start+250*ms))[0]]==k)[0].size
        n[k] = np.where(i[np.where(np.logical_and(t>=t_response_end-250*ms, t<t_response_end))[0]]==k)[0].size
        n_react[k] = np.where(i[np.where(np.logical_and(t>=(t_reig_end), t<(t_reig_end+250*ms)))[0]]==k)[0].size
        n2[k] = np.where(i[np.where(np.logical_and(t>=sim_time-250*ms, t<sim_time))[0]]==k)[0].size

        # area2
        n_area2[k] = np.where(i2[np.where(np.logical_and(t2 >= t_response_end - 250 * ms, t2 < t_response_end))[0]] == k)[0].size
        n_react_area2[k] = np.where(i2[np.where(np.logical_and(t2 >= (t_reig_end), t2 < (t_reig_end+250*ms)))[0]] == k)[0].size
        n2_area2[k] = np.where(i2[np.where(np.logical_and(t2 >= sim_time - 250 * ms, t2 < sim_time))[0]] == k)[0].size

    # angle/strength for center of first trial bump, angle
    bump_center, R1_A1 = decode(n, N_e)
    reactivation, R_react = decode(n_react, N_e)
    bump_center2, R2_A1 = decode(n2, N_e)

    # angle/strength for center of first trial bump, angle
    bump_center_area2, R1_A2 = decode(n_area2, N_e)
    reactivation_area2, R_react2 = decode(n_react_area2, N_e)
    bump_center2_area2, R2_A2 = decode(n2_area2, N_e)

    monitor = {'i': [i, i2], 't': [t, t2], \
               'depr': [depr, depr2], 'facil': [facil, facil2],\
               'stp': [stp_val, stp_val2]}

    ####################################################################################
    #                                  SET FIGURE PARAMETERS                           #
    ####################################################################################

    areas = range(2)
    ######################################################################################################
    #                                                 PLOTS                                              #
    ######################################################################################################
    # x = np.linspace(-(sim_time-250*ms), 250*ms, len(t))
    # # #
    #
    # # # create stimulus plot
    # xx = np.arange(0, sim_time, 0.1)
    # timeIdx_stim = np.where(xx > 0.5)[0]
    # stim = np.zeros((xx.shape))
    # stim[int(t_stimulus1_start / (100 * ms)):int(t_stimulus1_end / (100 * ms))] = stimulus_strength
    # stim[
    # int(t_response_end / (100 * ms)):int((t_response_end + t_stimulus_duration) / (100 * ms))] = -2 * stimulus_strength
    # #stim[int(t_reig_start / (100 * ms)):int(t_reig_end / (100 * ms))] = reig_strength
    # #stim[int(t_reig_end / (100 * ms) + 2):int(t_reig_end / (100 * ms) + 6)] = -stimulus_strength
    # #stim[int(t_stimulus2_start / (100 * ms)):int(t_stimulus2_end / (100 * ms))] = stimulus_strength
    #
    # #
    # f, ax = plt.subplots(len(areas), 1, figsize=(2.2, 1.9), sharex=True, \
    #                  gridspec_kw={'height_ratios': [0.5, 0.5]})
    # sns.despine()
    # for ar, area in enumerate(list(areas)):
    #     time_idx = np.where(monitor['t'][area]>500*ms)[0]
    #     #im = ax[ar + 1].imshow(np.append(monitor['stp'][area][int(3*N_e/4):], monitor['stp'][area][:int(3*N_e/4)], axis=0),\
    #     #                       aspect='auto', origin='lower', \
    #     #                    extent=[0, (sim_time / ms) / 1000, 0, N_e], cmap='YlOrRd')  # cividis
    #     ax[ar].plot(monitor['t'][area][time_idx]-t_stimulus1_start, (monitor['i'][area][time_idx]+N_e/4)%N_e, 'k,', ms=3)  # t
    #     ax[ar].plot(-0.2, (stim1A_location+90)%360*N_e/360, 'kv', ms=3)  # t
    #     ax[ar].plot(-0.2, (stim1B_location+90)%360*N_e/360, 'kD', ms=3)  # t
    #     ax[ar].fill_between([0, t_stimulus1_end-t_stimulus1_start], [0, 0], [N_e, N_e], color='grey', alpha=.3)
    #     #ax[ar + 1].fill_between([t_stimulus2_start, t_stimulus2_end], [0, 0], [N_e, N_e], color='grey', alpha=.3)
    #     #ax[ar + 1].set_ylim([0, N_e])
    #     #ax[ar + 1].set_ylabel('RF ($^\circ$)')
    #     if ar == 0:
    #         ax[ar].set_ylabel('A$_{Left}$')
    #     else:
    #         ax[ar].set_ylabel('A$_{Right}$')
    #     ax[ar].set_yticks([N_e / 4, N_e / 2, 3*N_e / 4])
    #     ax[ar].set_yticklabels([0, 90, 180])
    #     ax[ar].axhline(N_e/2, color='k', dashes=[1,1], alpha=0.5)
    #     ax[ar].yaxis.set_ticks_position('left')
    #     #ax[ar + 1].set_ylim([N_e / 2 - N_e / 4, N_e / 2 + N_e / 4])
    # # if REACT1 == True:
    # #     ax[0].fill_between([t_reig_start, t_reig_end], [0, 0], [N_e, N_e], color='yellow', alpha=.3)
    # # if REACT2 == True:
    # #     ax[1].fill_between([t_reig_start, t_reig_end], [0, 0], [N_e, N_e], color='yellow', alpha=.3)
    # ax[1].xaxis.set_ticks_position('bottom')
    # ax[1].set_xlabel('time (s)')
    # plt.subplots_adjust(wspace=0, hspace=0.25)
    # ax[1].set_xticks([0,2,4])
    # plt.tight_layout()
    # if HEMI_A == HEMI_B:
    #     plt.savefig('./Trial_Multiitem_Same.svg')
    # else:
    #     plt.savefig('./Trial_Multiitem_Opposite.svg')
    # plt.show()

    # f, ax = plt.subplots(2, 1, figsize=(3., 2.4), sharex=True, sharey=True)
    # cbar = f.colorbar(im)
    # cbar.set_label('STP')
    # plt.tight_layout()
    # # # plt.savefig('TwoAreaModel_FullSimulation_colorbar.svg')
    # plt.show()

    # # #
    # # # plot facil*depression
    # plt.figure()
    # plt.imshow(monitor['stp'][area], aspect='auto', origin='lower', \
    #            extent=[0, (sim_time / ms) / 1000, 0, N_e], cmap='YlOrRd_r')
    # plt.colorbar()
    # plt.show()
    # #
    # # # plot facilitation at stim1, depression at stim1
    # plt.figure()
    # neuron_range = range(int((stim1A_location-10)*N_e/360), int((stim1A_location+10)*N_e/360))
    # plt.plot(np.mean(monitor['facil'][area][neuron_range], axis=0), color='darkorange')
    # plt.plot(np.mean(monitor['depr'][area][neuron_range], axis=0), color='k')
    # plt.plot(np.mean(monitor['stp'][area][neuron_range], axis=0), color='yellow')
    # plt.show()
    #######################################################################################################
    #                                   CREATE AN ANIMATION                                               #
    #######################################################################################################
    from matplotlib.animation import FuncAnimation
    from matplotlib import animation
    from matplotlib.animation import FuncAnimation
    from matplotlib import animation
    import pandas as pd

    # initializing a figure in
    # which the graph will be plotted
    framesteps = 100 / 1000  # s
    moving_window = 200 / 1000
    frames = np.arange(0, (sim_time / ms) / 1000 + framesteps - moving_window, framesteps)
    neuronaxis = np.arange(0, N_e, 10)
    neuronbincenters = [(neuronaxis[i] + neuronaxis[i + 1]) / 2. for i in range(len(neuronaxis) - 1)]

    bump_avg = [[], []]
    bump_loc = [[], []]
    moving_fr = [[], []]
    moving_fr_center = [[], []]
    for area in areas:
        # compute moving average
        moving_average = []
        moving_sum = []
        for w1 in frames:
            idx = (monitor['t'][area] / (ms * 1000) >= w1) & \
                  (monitor['t'][area] / (ms * 1000) < w1 + moving_window)
            moving_average.append(circmean(monitor['i'][area][idx], low=0, high=N_e))
            moving_sum.append(np.histogram(monitor['i'][area][idx], bins=neuronaxis)[0])
            moving_fr[area].append(len(monitor['i'][area][idx]) / (moving_window * N_e))
            center_neurons = 200
            moving_fr_center[area].append(
                len(np.where(abs(monitor['i'][area][idx] - stim1A_location / 360 * N_e) < center_neurons / 2)[0]) / \
                (moving_window * center_neurons))

        # save bump drift
        moving_avg_nonan = np.array(moving_average)
        # replace non-decodable locations (no spiking in window) with a random location
        nan_idx = np.where(np.isnan(moving_average))[0]
        moving_avg_nonan[nan_idx] = np.random.randint(0, N_e, len(nan_idx))
        bump_avg[area] = [moving_avg_nonan]
        # convert neural bumpcenter location to angle in radians in [-pi, pi]
        bump_loc[area] = moving_avg_nonan * (2 * np.pi / N_e) - (
                moving_avg_nonan * (2 * np.pi / N_e) > np.pi) * 2 * np.pi

    # plt.axhline(stim1A_location * N_e / 360, color='k')
    # plt.axhline(stim2_location * N_e / 360, color='r')
    # for area in areas:
    #     plt.plot(bump_avg[area][0])
    # plt.show()
    #
    # dict = {'time': [list(frames)], 'bump_average_area1': bump_avg[0], 'bump_average_area2': bump_avg[1],\
    #         'trial2start': t_stimulus2_start/(ms*1000)}
    # df = pd.DataFrame(dict)
    # try:
    #     orig = pd.read_csv('TwoAreaBumpdrift_consecutiveReactivations.csv')
    #     df = orig.append(df)
    # except FileNotFoundError:
    #     print('Creating new file')
    # df.reset_index(drop=True, inplace=True)
    # df.to_csv('Bumpdrift_consecutiveReactivations.csv')

    #################################################################################################
    #                                    ONE AREA PLOT                                             #
    #################################################################################################

    # sns.set_theme()
    # sns.set_style("white")
    #
    # # #paper
    # sns.set_context("paper")
    # params = {'legend.fontsize': 11,
    #           'figure.figsize': (7.8, 2.2),
    #           'axes.labelsize': 12,
    #           'axes.titlesize': 14,
    #           'xtick.labelsize': 11,
    #           'ytick.labelsize': 11}
    # pylab.rcParams.update(params)
    #
    # f, ax = plt.subplots(2,1,figsize=(2.8, 2.4), sharex=True, sharey=True)
    # #f.subplots_adjust(wspace=-1.4)
    # for area in areas:
    #     # plot STP
    #     im = ax[area].imshow(monitor['stp'][area], aspect='auto', origin='lower', \
    #                extent=[-(t_stimulus1_start/ms)/1000, ((sim_time-t_stimulus1_start) / ms) / 1000, 0, N_e], cmap='YlOrRd')  # cividis
    #     # plot spikes
    #     ax[area].plot(monitor['t'][area]-t_stimulus1_start, (monitor['i'][area]), 'k,', ms=3)  # t
    #     # mark stimulus and times
    #     ax[area].scatter(-150*ms, stim1A_location*N_e/360, color='w', marker='>', s=30)
    #     ax[area].scatter(t_stimulus2_start-t_stimulus1_start-150*ms, stim2_location*N_e/360, color='w', marker='>', s=30)
    #     ax[area].fill_between([0, t_stimulus1_end-t_stimulus1_start],[0,0], [N_e, N_e], color='k', alpha=.3)
    #     ax[area].fill_between([t_stimulus2_start-t_stimulus1_start, t_stimulus2_end-t_stimulus1_start],[0,0], [N_e, N_e], color='k', alpha=.3)
    #     ax[area].fill_between([t_reig_start-t_stimulus1_start,t_reig_end-t_stimulus1_start], [0,0], [N_e,N_e], color='saddlebrown', alpha=.3)
    #     ax[area].set_yticks([400, 600])
    #     ax[area].set_yticklabels([int(400/N_e*360), int(600/N_e*360)])
    # ax[area].set_ylim([N_e/4, 240*N_e/360])
    # ax[1].set_xlabel('time (s)')
    # f.text(0, 0.6, 'neuron label ($^\circ$)', va='center', rotation='vertical', fontsize=11)
    # plt.tight_layout()
    # plt.savefig('TwoAreaModel_FullSimulation.svg')
    # plt.savefig('TwoAreaModel_FullSimulation.png')
    # plt.show()
    #
    # f, ax = plt.subplots(2, 1, figsize=(3., 2.4), sharex=True, sharey=True)
    # cbar = f.colorbar(im)
    # cbar.set_label('STP')
    # plt.tight_layout()
    # plt.savefig('TwoAreaModel_FullSimulation_colorbar.svg')
    # plt.show()

    #######################################################################################################
    #                                                 SAVE STUFF                                          #
    #######################################################################################################

    # with open(stp_log, 'a') as myfile:
    #    myfile.write(str(trial1)+" "+str(trial2)+" "+str(t_delay1_duration/ms)+" "+str(t_delay2_duration/ms)+" "+str(bump_center)+" "+str(bump_center2)+" "+' '.join(map(str,r_full))+'\n')

    dict = {'stim1A_location': stim1A_location, 'stim1B_location': stim1B_location, \
            'HEMI_A':HEMI_A, 'HEMI_B':HEMI_B, 'HEMI_DIFF': ['within' if HEMI_A==HEMI_B else 'across'][0],\
            'FR_area1': [moving_fr[0]], 'FR_area2': [moving_fr[1]], \
            'FR_center1_area1': [moving_fr_center[0]], 'FR_center1_area2': [moving_fr_center[1]], \
            'response1bump_area1': [n], 'response1_area1': bump_center, 'responsestrength1_area1': R1_A1, \
            'response1bump_area2': [n_area2],'response1_area2': bump_center_area2, 'responsestrength1_area2': R1_A2, \
            'delay1': t_delay1_duration / ms,\
            'stim_start': t_stimulus1_start / ms,\
            'stim_duration': t_stimulus_duration / ms, \
            'bump_average_area1': [bump_loc[0]], 'bump_average_area2': [bump_loc[1]], \
            'time_average': [list(frames)], \
            'window': moving_window, 'steps': framesteps, 'across_factor': across_factor}  # ,\

    #
    # 'stp_area1': [stp_val], 'stp_area2':[stp_val2]}
    df = pd.DataFrame(dict)
    df['FR_area1'] = df['FR_area1'].map(list)
    df['FR_area2'] = df['FR_area2'].map(list)
    df['response1bump_area1'] = df['response1bump_area1'].map(list)
    df['response1bump_area2'] = df['response1bump_area2'].map(list)
    df['FR_center1_area1'] = df['FR_center1_area1'].map(list)
    df['FR_center1_area2'] = df['FR_center1_area2'].map(list)
    df['bump_average_area1'] = df['bump_average_area1'].map(list)
    df['bump_average_area2'] = df['bump_average_area2'].map(list)
    df['time_average'] = df['time_average'].map(list)
    try:
        df.to_csv(filename + '.csv', mode='a', header=False)
        # orig = pd.read_pickle(filename+'.pickle')
        # df = orig.append(df)
    except FileNotFoundError:
        print('Creating new file')
        df.to_csv(filename + '.csv')

#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################

sims = 1501
across_factor = 0.002
simulations = np.ones((sims))*across_factor
with mp.Pool(numcores) as p:
    print('Starting '+str(sims)+' simulations')
    t0 = time.time()
    results = p.map(run_simulation, simulations)
    t1 = time.time()
    print('Used time: ' + str(t1 - t0))
    time.sleep(0.1) # otherwise all trials try to write at the same time


# for simus in range(1):
#     t0 = time.time()
#     run_simulation(0.002)
#     t1 = time.time()
#     print('Used time: '+str(t1-t0))
