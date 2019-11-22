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
    params['net_record_spikes'] = 25  # 200 # number of neurons to record spikes from
    params['net_record_example_v_traces'] = 0.  # number of recorded v_traces, if no trace should be computed: 0
    params['net_record_all_neurons'] = False
    params['net_record_all_neurons_dt'] = 10.  # keep this high; otherwise great deal of memory, zero if not
    params['net_record_v_stats'] = True  # mean and std
    params['net_record_w_stats'] = True  # mean and std
    params['net_record_w'] = 10  # 0000 # record 100 w traces
    params['net_record_dt'] = 1.  # [ ms]
    params['net_w_refr'] = True  # clamp (True) or don't clamp (False) w during the refractory period
    params['net_w_init'] = 0.  # 20. # [pA]
    params['net_v_lower_bound'] = None  # -100. #if None --> NO lower bound, else value of lower bound e.g. -200 in [mV]
    params['connectivity_type'] = 'fixed'  # 'binomial'#'fixed'
    # initial conditions
    # the transient response of the network and the solution of the fp eq are in good agreement for the init: normal
    # note for uniform distribution: by default the uniform initial distribution is set on the interval [Vr, Vcut]
    params['net_v_init'] = 'normal'  # 'uniform' #'normal', 'delta'
    params['net_delta_peak'] = -70.
    params['net_normal_mean'] = -100.  # mV
    params['net_normal_sigma'] = 10.  # mV
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
    params['t_ref_exc'] = 0.0  # [ms]

    # todo: adjust params for inhibitory populations
    params['C_inh1'] = 200.  # [pF]
    params['gL_inh1'] = 10.  # [nS]
    params['taum_inh1'] = params['C_inh1'] / params['gL_inh1']  # [ms]
    params['EL_inh1'] = -65.  # [mV] # reversal potential for membrane potential v
    params['Ew_inh1'] = -80.  # [mV] # reversal potential for adaptation param w
    params['VT_inh1'] = -50.  # [mV]
    params['deltaT_inh1'] = 1.5  # [mV]
    params['Vcut_inh1'] = -40.  # [mV]
    params['tauw_inh1'] = 200.  # [ms]
    params['a_inh1'] = 4.  # [nS]              subthreshold adaptation param
    params['b_inh1'] = 40.  # [pA]              spike-frequency adaptation param
    params['Vr_inh1'] = -70.  # [mV]
    params['t_ref_inh1'] = 0.0  # [ms]

    params['C_inh2'] = 200.  # [pF]
    params['gL_inh2'] = 10.  # [nS]
    params['taum_inh2'] = params['C_inh2'] / params['gL_inh2']  # [ms]
    params['EL_inh2'] = -65.  # [mV] # reversal potential for membrane potential v
    params['Ew_inh2'] = -80.  # [mV] # reversal potential for adaptation param w
    params['VT_inh2'] = -50.  # [mV]
    params['deltaT_inh2'] = 1.5  # [mV]
    params['Vcut_inh2'] = -40.  # [mV]
    params['tauw_inh2'] = 200.  # [ms]
    params['a_inh2'] = 4.  # [nS]              subthreshold adaptation param
    params['b_inh2'] = 40.  # [pA]              spike-frequency adaptation param
    params['Vr_inh2'] = -70.  # [mV]
    params['t_ref_inh2'] = 0.0  # [ms]

    # Network size
    params['N_e'] = 10000
    params['N_i1'] = 2500
    params['N_i2'] = 2500

    # for recurrency
    # todo: adjust coupling parameters
    params['J_etoe'] = .01  # [mV]
    params['J_etoi1'] = .01  # [mV]
    params['J_etoi2'] = .01  # [mV]
    params['J_i1toe'] = .01  # [mV]
    params['J_i2toe'] = .01  # [mV]
    params['J_i1toi1'] = .01  # [mV]
    params['J_i2toi2'] = .01  # [mV]
    params['J_i1toi2'] = .01  # [mV]
    params['J_i2toi1'] = .01  # [mV]
    params['K'] = 100  # number of connections
    params['K_etoe'] = 100  # number of connections
    params['K_etoi1'] = 100  # number of connections
    params['K_etoi2'] = 100  # number of connections
    params['K_i1toe'] = 100  # number of connections
    params['K_i2toe'] = 100  # number of connections
    params['K_i1toi1'] = 100  # number of connections
    params['K_i2toi2'] = 100  # number of connections
    params['K_i1toi2'] = 100  # number of connections
    params['K_i2toi1'] = 100  # number of connections

    # initial value for the mean adaptation current
    params['wm_init'] = 0.  # [pA]

    # for recurrency
    params['taud'] = 3.  # [ms]
    params['const_delay'] = 5.  # [ms]
    params['delay_type'] = 0  # options [0: no delay, 1: const, 2: exp, 3: const+exp]

    # for plotting
    # colors[modelname] = color
    params['color'] = {'net': 'b', 'fp': '0.6', 'ln_exp': 'darkmagenta', 'ln_dos': 'cyan',
                       'ln_bexdos': 'green', 'spec1': 'darkgreen',
                       'spec2_red': 'pink', 'spec2': 'orangered'}
    params['lw'] = {'net': '1', 'fp': '2', 'ln_exp': '1', 'ln_dos': '2', 'ln_bexdos': '2',
                    'spec1': '1', 'spec2_red': '1', 'spec2': '1'}
    return params
