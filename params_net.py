""" manages the parameter sets for all models. all parameter are given without brian units except the ones that
are used for brian2 simulations. So far, only brian2 model parameters are included. FP parameters will be added later.
"""


def get_params():
    params = dict()

    # runtime options #
    # reduced models

    # params for ou process #
    params['ou_stationary'] = True  # if False --> ou process starts at X0

    # parameters for network simulation
    params['net_record_spikes'] = 1000  # 200 # number of neurons to record spikes from
    params['net_record_all_neurons'] = True  # False
    params['net_record_all_neurons_dt'] = 1.  # keep this high; otherwise great deal of memory, zero if not
    params['net_record_dt'] = 1.  # [ ms]
    params['net_w_refr'] = True  # clamp (True) or don't clamp (False) w during the refractory period
    params['net_w_init_e'] = 0.  # 20. # [pA]
    params['net_w_init_i'] = 0.  # 20. # [pA]
    params['net_v_lower_bound'] = None  # -100. #if None --> NO lower bound, else value of lower bound e.g. -200 in [mV]
    # initial conditions
    # the transient response of the network and the solution of the fp eq are in good agreement for the init: normal
    # note for uniform distribution: by default the uniform initial distribution is set on the interval [Vr, Vcut]
    params['net_delta_peak_E'] = -70.
    params['net_delta_peak_I'] = -70.

    # standalone mode for network sim
    params['brian2_standalone'] = True
    params['brian2_device'] = 'cpp_standalone'
    # integration method for (should be specified for brian2_rc3)
    params['net_integration_method'] = 'heun'  # 'heun'

    # neuron model parameters (AdEX)

    # Excitatory cells
    params['C_exc'] = 200.  # [pF]
    params['gL_exc'] = 10.  # [nS]
    params['taum_exc'] = params['C_exc'] / params['gL_exc']  # [ms]
    params['EL_exc'] = -65.  # [mV] # reversal potential for membrane potential v
    params['Ew_exc'] = -80.  # [mV] # reversal potential for adaptation param w
    params['VT_exc'] = -50.  # [mV]
    params['deltaT_exc'] = 1.5  # [mV]
    params['Vcut_exc'] = -40.  # [mV]
    params['tauw_exc'] = 200.  # [ms]
    params['a_exc'] = 4.  # [nS]              subthreshold adaptation param
    params['b_exc'] = 40.  # [pA]              spike-frequency adaptation param
    params['Vr_exc'] = -70.  # [mV]
    params['t_ref_exc'] = 1.0  # [ms]

    # AMPA synapse (excitatory)
    params['tau_AMPA'] = 3.0  # [ms]
    params['E_AMPA'] = 0.0  # [mV] Reversal potential of AMPA synapse, usually set to 0 mV.

    # Inhibitory cells
    params['C_inh1'] = 200.  # [pF]
    params['gL_inh1'] = 10.  # [nS]
    params['taum_inh1'] = params['C_inh1'] / params['gL_inh1']  # [ms]
    params['EL_inh1'] = -65.  # [mV] # reversal potential for membrane potential v
    params['Ew_inh1'] = -80.  # [mV] # reversal potential for adaptation param w
    params['VT_inh1'] = -50.  # [mV]
    params['deltaT_inh1'] = 1.5  # [mV]
    params['Vcut_inh1'] = -40.  # [mV]
    params['tauw_inh1'] = 200.  # [ms]
    params['a_inh1'] = 0.  # [nS]              subthreshold adaptation param
    params['b_inh1'] = 0.  # [pA]              spike-frequency adaptation param
    params['Vr_inh1'] = -70.  # [mV]
    params['t_ref_inh1'] = 1.0  # [ms]

    # GABA synapse (inhibitory)
    params['tau_GABA'] = 9.0  # [ms]
    params['E_GABA'] = -70.0  # [mV] Reversal potential of GABA synapse , usually set to -70 to -75mV

    # Network size
    params['N_e'] = 1000
    params['N_i'] = 250

    # for recurrency
    factor = 10.
    params['J_etoe'] = 0.01 * factor  # [nS] synaptic strength E-E conns within population
    params['J_etoi'] = .05 * factor  # .25      # [nS] synaptic strength E-I conns within population
    params['J_itoe'] = 1. * factor  # [nS] synaptic strength I-E conns within population
    params['J_itoi'] = 0.015 * factor  # [nS] synaptic strength I-I conns within population

    params['J_ppee'] = 0.5  # .1      # [nS] synaptic strength E-E conns between population
    params['J_ppei'] = 0.  # .1      # [nS] synaptic strength E-I conns between population

    params['K_etoe'] = 100  # number of E-E connections within population
    params['K_etoi'] = 100  # number of E-I connections within population
    params['K_itoe'] = 100  # number of I-E connections within population
    params['K_itoi'] = 100  # number of I-I connections within population

    params['K_ppee'] = 10  # number of E-E connections between population
    params['K_ppei'] = 10  # number of E-I connections between population

    # initial value for the mean adaptation current
    params['wm_init'] = 0.  # [pA]

    # for recurrency
    params['const_delay'] = 0.  # [ms]

    # for plotting
    # colors[modelname] = color
    params['color'] = {'net': 'b'}
    params['lw'] = {'net': '1'}
    return params
