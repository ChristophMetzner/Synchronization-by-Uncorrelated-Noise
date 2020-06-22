"""
Manages the parameter sets for all models.

All parameter are given without brian units except the ones that are used for brian2 simulations.
So far, only brian2 model parameters are included. FP parameters will be added later.
"""


def get_params():
    params = dict()

    # params for poisson input
    params["poisson_enabled"] = [False, False]
    params["poisson_mean_input"] = 200
    params["poisson_size"] = 200
    params["poisson_variance"] = 0.9
    params["poisson_p"] = 0.83

    # params for ou process #
    params["ou_enabled"] = [True, True]
    params["ou_stationary"] = True  # if False --> ou process starts at X0

    # parameters for network simulation
    params["net_record_spikes"] = 1000  # 200 # number of neurons to record spikes from
    params["net_record_all_neurons"] = True  # False

    # keep this high; otherwise great deal of memory, zero if not
    params["net_record_all_neurons_dt"] = 1.0
    params["net_record_dt"] = 1.0  # [ ms]

    # clamp (True) or don't clamp (False) w during the refractory period
    params["net_w_refr"] = True

    # 20. # [pA]
    params["net_w_init_e"] = 0.0
    # 20. # [pA]
    params["net_w_init_i"] = 0.0

    # -100. #if None --> NO lower bound, else value of lower bound e.g. -200 in [mV]
    params["net_v_lower_bound"] = None

    # initial conditions
    # the transient response of the network and the solution of the fp eq are in good agreement for the init: normal
    # note for uniform distribution: by default the uniform initial distribution is set on the interval [Vr, Vcut]
    params["net_delta_peak_E"] = -70.0
    params["net_delta_peak_I"] = -70.0

    # standalone mode for network sim
    params["brian2_standalone"] = False
    params["brian2_device"] = "cpp_standalone"
    # Set to numpy if problems with cython occur
    params["brian2_codegen_target"] = "cython"
    # integration method for (should be specified for brian2_rc3)
    params["net_integration_method"] = "heun"  # 'heun'

    # neuron model parameters (AdEX)

    # Excitatory cells
    params["C_exc"] = 200.0  # [pF] Capacitance
    params["gL_exc"] = 10.0  # [nS] Conductance
    params["taum_exc"] = params["C_exc"] / params["gL_exc"]  # [ms]
    params["EL_exc"] = -65.0  # [mV] # reversal potential for membrane potential v
    params["Ew_exc"] = -80.0  # [mV] # reversal potential for adaptation param w
    params["VT_exc"] = -50.0  # [mV]
    params["deltaT_exc"] = 1.5  # [mV]
    params["Vcut_exc"] = -40.0  # [mV]
    params["tauw_exc"] = 200.0  # [ms]
    params["a_exc"] = 4.0  # [nS]              subthreshold adaptation param
    params["b_exc"] = 40.0  # [pA]              spike-frequency adaptation param
    params["Vr_exc"] = -70.0  # [mV]
    params["t_ref_exc"] = 1.0  # [ms]

    # AMPA synapse (excitatory)
    params["tau_AMPA"] = 3.0  # [ms]
    # [mV] Reversal potential of AMPA synapse, usually set to 0 mV.
    params["E_AMPA"] = 0.0

    # Inhibitory cells
    params["C_inh1"] = 200.0  # [pF]
    params["gL_inh1"] = 10.0  # [nS]
    params["taum_inh1"] = params["C_inh1"] / params["gL_inh1"]  # [ms]
    params["EL_inh1"] = -65.0  # [mV] # reversal potential for membrane potential v
    params["Ew_inh1"] = -80.0  # [mV] # reversal potential for adaptation param w
    params["VT_inh1"] = -50.0  # [mV]
    params["deltaT_inh1"] = 1.5  # [mV]
    params["Vcut_inh1"] = -40.0  # [mV]
    params["tauw_inh1"] = 200.0  # [ms]
    params["a_inh1"] = 0.0  # [nS]              subthreshold adaptation param
    params["b_inh1"] = 0.0  # [pA]              spike-frequency adaptation param
    params["Vr_inh1"] = -70.0  # [mV]
    params["t_ref_inh1"] = 1.0  # [ms]

    # GABA synapse (inhibitory)
    # Neuronal Dynamics (https://neuronaldynamics.epfl.ch/online/Ch3.S1.html#SS1.p3) suggests 6 ms.
    # [ms] # 9.0 starting point
    params["tau_GABA"] = 6.0

    # [mV] Reversal potential of GABA synapse , usually set to -70 to -75mV
    params["E_GABA"] = -70.0

    # Network size
    params["N_pop"] = 1  # number of populations 1 or 2
    params["N_e"] = 1000
    params["N_i"] = 250

    # for recurrency
    factor = 10.0

    # [nS] synaptic strength E-E conns within population
    params["J_etoe"] = 0.01 * factor

    # [nS] synaptic strength E-I conns within population
    # .25
    params["J_etoi"] = 0.05 * factor

    # [nS] synaptic strength I-E conns within population
    params["J_itoe"] = 1.0 * factor

    # [nS] synaptic strength I-I conns within population
    params["J_itoi"] = 0.3 * factor

    # [nS] synaptic strength E-E conns between population
    # .1
    params["J_ppee"] = 0.2

    # [nS] synaptic strength E-I conns between population
    # .1
    params["J_ppei"] = 0.1

    # connectivity inside populations E <-> I
    params["p_etoe"] = 0.4
    params["p_etoi"] = 0.4
    params["p_itoe"] = 0.1
    params["p_itoi"] = 0.1

    # connectivity between populations
    params["p_ppee"] = 0.01
    params["p_ppei"] = 0.01

    # initial value for the mean adaptation current
    params["wm_init"] = 0.0  # [pA]

    # for recurrency
    params["const_delay"] = 0.1  # [ms]

    # for plotting
    # colors[modelname] = color
    # TODO: do we need this?
    # params["color"] = {"net": "b"}
    # params["lw"] = {"net": "1"}
    return params
