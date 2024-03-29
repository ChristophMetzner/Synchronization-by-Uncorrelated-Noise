from __future__ import print_function
from brian2 import *
import time
import os
import shutil
import gc
from utils_net import fixed_connectivity

cpp_default_dir = 'brian2_compile'


def network_sim(signal, params):

    if params['brian2_standalone']:
        # build on run = False (changed for brian2_rc3)
        set_device(params['brian2_device'], build_on_run=False)
        device.insert_code('main', 'srand('+str(int(time.time())+os.getpid())+');')
        # standalonedir = 'standalone_{}_pid{}'.format(time.strftime("%Y-=%m-%dT%H:%M:%S"), os.getpid())
    else:
        prefs.codegen.target = 'numpy'

    # stripe on brian units within the copied dict for the simulation so that brian can work with them

    # Neuron group specific parameters

    # Excitatory cells
    C_e = params['C_exc']*pF
    gL_e = params['gL_exc']*nS
    taum_e = params['taum_exc']*ms
    EL_e = params['EL_exc']*mV
    Ew_e = params['Ew_exc']*mV
    VT_e = params['VT_exc']*mV
    negVT_e = -VT_e
    deltaT_e = params['deltaT_exc']*mV
    Vcut_e = params['Vcut_exc']*mV
    tauw_e = params['tauw_exc']*ms
    Vr_e = params['Vr_exc']*mV
    t_ref_e = params['t_ref_exc']*ms

    # Inhibitory cells
    C_i = params['C_inh1']*pF
    gL_i = params['gL_inh1']*nS
    taum_i = params['taum_inh1']*ms
    EL_i = params['EL_inh1']*mV
    Ew_i = params['Ew_inh1']*mV
    VT_i = params['VT_inh1']*mV
    negVT_i = -VT_i
    deltaT_i = params['deltaT_inh1']*mV
    Vcut_i = params['Vcut_inh1']*mV
    tauw_i = params['tauw_inh1']*ms
    Vr_i = params['Vr_inh1']*mV
    t_ref_i = params['t_ref_inh1']*ms



    # general parameters
    dt_sim = params['net_dt'] * ms
    net_w_init_e = params['net_w_init_e']*pA
    net_w_init_i = params['net_w_init_i']*pA
    rates_dt = params['net_record_dt'] *ms
    runtime = params['runtime']*ms
    time_r = np.arange(0., runtime/ms, rates_dt/ms)
    time_v = np.arange(0., runtime/ms, dt_sim/ms)

    # ratio used for smoothing the spike histogram to match resolution
    ratesdt_dtsim = int(rates_dt/dt_sim)
    N_e = params['N_e']
    N_i = params['N_i']

    # garbage collect so we can run multiple brian2 runs
    gc.collect()

    # seed our random number generator!  necessary on the unix server
    np.random.seed()

    mu_ext_array_e = signal[0]  # [mV/ms]
    sigma_ext_array_e = signal[1]  # [mV/sqrt(ms)]
    # todo: Change runmodel.py to create specific input for inhibitory population(s)
    mu_ext_array_i = np.zeros_like(mu_ext_array_e) #signal[0] # [mV/ms]
    sigma_ext_array_i = np.zeros_like(mu_ext_array_e) # signal[1] # [mV/sqrt(ms)]


    # what is recorded during the simulation
    record_spikes = params['net_record_spikes']
    record_all_v_at_times = True if params['net_record_all_neurons'] else False

    # simulation time step
    simclock = Clock(dt_sim)

    w_refr_e = ' (unless refractory)' if 'net_w_refr_e' not in params or params['net_w_refr_e'] else ''
    w_refr_i = ' (unless refractory)' if 'net_w_refr_i' not in params or params['net_w_refr_i'] else ''


    a_e = params['a_exc']
    b_e = params['b_exc']
    # convert to array if adapt params are scalar values
    if type(a_e) in [int ,float]:
        a_e = np.ones_like(mu_ext_array_e)*a_e
    if type(b_e) in [int, float]:
        b_e = np.ones_like(mu_ext_array_e)*b_e

    a_i = params['a_inh1']
    b_i = params['b_inh1']
    # convert to array if adapt params are scalar values
    if type(a_i) in [int, float]:
        a_i = np.ones_like(mu_ext_array_e)*a_i
    if type(b_i) in [int, float]:
        b_i = np.ones_like(mu_ext_array_e)*b_i



    # decide if there's adaptation
    have_adap_e = True if (a_e.any() > 0.0) or (b_e.any() > 0.0) else False
    have_adap_i = True if (a_i.any() > 0.0) or (b_i.any() > 0.0) else False


    # convert numpy arrays to TimedArrays
    a_e = TimedArray(a_e*nS, dt_sim)
    b_e = TimedArray(b_e*pA, dt_sim)
    a_i = TimedArray(a_i*nS, dt_sim)
    b_i = TimedArray(b_i*pA, dt_sim)


    # transform the external input into TimedArray
    mu_ext_e = TimedArray(mu_ext_array_e*(mV/ms), dt_sim)
    sigma_ext_e = TimedArray(sigma_ext_array_e*(mV/sqrt(ms)), dt_sim)
    mu_ext_i = TimedArray(mu_ext_array_i*(mV/ms), dt_sim)
    sigma_ext_i = TimedArray(sigma_ext_array_i*(mV/sqrt(ms)), dt_sim)


    # get the model specific term EIF

    model_term_e = '((EL_e - v) + deltaT_e * exp((negVT_e + v) / deltaT_e)) / taum_e'
    model_term_i = '((EL_i - v) + deltaT_i * exp((negVT_i + v) / deltaT_i)) / taum_i'


    model_eqs_e = '''
        dv/dt = %s %s + mu_ext_e(t) + sigma_ext_e(t) * xi  : volt (unless refractory)
        %s
        ''' % (model_term_e, '- (w / C_e)' if have_adap_e else '',
               ('dw/dt = (a_e(t) * (v - Ew_e) - w) / tauw_e : amp %s' % w_refr_e) if have_adap_e else '')
    model_eqs_i = '''
       dv/dt = %s %s + mu_ext_i(t) + sigma_ext_i(t) * xi  : volt (unless refractory)
        %s
        ''' % (model_term_i, '- (w / C_i)' if have_adap_i else '',
               ('dw/dt = (a_i(t) * (v - Ew_i) - w) / tauw_i : amp %s' % w_refr_i) if have_adap_i else '')
    #model_eqs_i = '''
    #        dv/dt = %s %s  : volt (unless refractory)
    #        %s
    #        ''' % (model_term_i, '- (w / C_i)' if have_adap_i else '',
    #               ('dw/dt = (a_i(t) * (v - Ew_i) - w) / tauw_i : amp %s' % w_refr_i) if have_adap_i else '')


    # initialize Neuron groups

    # Population 1
    E = NeuronGroup(N=N_e, model=model_eqs_e,
                    threshold='v > Vcut_e', clock=simclock,
                    reset='v = Vr_e%s' % ('; w += b_e(t)' if have_adap_e else ''),
                    refractory=t_ref_e, method=params['net_integration_method'])

    I = NeuronGroup(N=N_i, model=model_eqs_i,
                     threshold='v > Vcut_i', clock=simclock,
                     reset='v = Vr_i%s' % ('; w += b_i(t)' if have_adap_i else ''),
                     refractory=t_ref_i, method=params['net_integration_method'])

    # Population 2
    E2 = NeuronGroup(N=N_e, model=model_eqs_e,
                    threshold='v > Vcut_e', clock=simclock,
                    reset='v = Vr_e%s' % ('; w += b_e(t)' if have_adap_e else ''),
                    refractory=t_ref_e, method=params['net_integration_method'])

    I2 = NeuronGroup(N=N_i, model=model_eqs_i,
                     threshold='v > Vcut_i', clock=simclock,
                     reset='v = Vr_i%s' % ('; w += b_i(t)' if have_adap_i else ''),
                     refractory=t_ref_i, method=params['net_integration_method'])

    # initialize PopulationRateMonitor
    rate_monitor_e = PopulationRateMonitor(E, name='aeif_ratemon_e')
    rate_monitor_e2 = PopulationRateMonitor(E2, name='aeif_ratemon_e2')
    rate_monitor_i = PopulationRateMonitor(I, name='aeif_ratemon_i')
    rate_monitor_i2 = PopulationRateMonitor(I2, name='aeif_ratemon_i2')


    # initialize net
    Net = Network(E, E2, I, I2, rate_monitor_e, rate_monitor_e2, rate_monitor_i, rate_monitor_i2)

    print('building synapses...')
    start_synapses = time.time()

    J_etoe = params['J_etoe']*mV
    J_etoi = params['J_etoi']*mV
    J_itoe = params['J_itoe']*mV
    J_itoi = params['J_itoi']*mV

    J_ppee = params['J_ppee']*mV
    J_ppei = params['J_ppei']*mV

    K_etoe = params['K_etoe']
    K_etoi = params['K_etoi']
    K_itoe = params['K_itoe']
    K_itoi = params['K_itoi']

    K_ppee = params['K_ppee']
    K_ppei = params['K_ppei']


    # Connections within population 1

    # synapses object
    # this only specifies the dynamics of the synapses. they get actually created when the .connect method is called
    SynEE = Synapses(E, E, on_pre='v+=J_etoe')
    sparsity = float(K_etoe)/N_e
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_e, K_etoe)
    SynEE.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynEE)

    SynEI = Synapses(E, I, on_pre='v+=J_etoi')
    sparsity = float(K_etoi)/N_i
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_etoi)
    SynEI.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynEI)


    SynIE = Synapses(I, E, on_pre='v+=J_itoe')
    sparsity = float(K_itoe)/N_e
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_itoe)
    SynIE.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynIE)

    SynII = Synapses(I, I, on_pre='v+=J_itoi')
    sparsity = float(K_itoi)/N_i
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_itoi)
    SynII.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynII)



    # Connections within population 2

    SynEE2 = Synapses(E2, E2, on_pre='v+=J_etoe')
    sparsity = float(K_etoe) / N_e
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_e, K_etoe)
    SynEE2.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynEE2)

    SynEI2 = Synapses(E2, I2, on_pre='v+=J_etoi')
    sparsity = float(K_etoi) / N_i
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_etoi)
    SynEI2.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynEI2)

    SynIE2 = Synapses(I2, E2, on_pre='v+=J_itoe')
    sparsity = float(K_itoe) / N_e
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_itoe)
    SynIE2.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynIE2)

    SynII2 = Synapses(I2, I2, on_pre='v+=J_itoi')
    sparsity = float(K_itoi) / N_i
    assert 0 <= sparsity <= 1.0
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_itoi)
    SynII2.connect(i=prelist, j=postlist)
    # delays
    # no delay; nothing has to be implemented
    Net.add(SynII2)


    # Connections between populations

    SynE1E2 = Synapses(E, E2, on_pre='v+=J_ppee')
    # connectivity type
    prelist, postlist = fixed_connectivity(N_e, K_ppee)
    SynE1E2.connect(i=prelist, j=postlist)
    # delays
    SynE1E2.delay = '{} * ms'.format(params['const_delay'])
    Net.add(SynE1E2)

    SynE2E1 = Synapses(E2, E, on_pre='v+=J_ppee')
    # connectivity type
    prelist, postlist = fixed_connectivity(N_e, K_ppee)
    SynE2E1.connect(i=prelist, j=postlist)
    # delays
    SynE2E1.delay = '{} * ms'.format(params['const_delay'])
    Net.add(SynE2E1)

    SynE1I2 = Synapses(E, I2, on_pre='v+=J_ppei')
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_ppei)
    SynE1I2.connect(i=prelist, j=postlist)
    # delays
    SynE1I2.delay = '{} * ms'.format(params['const_delay'])
    Net.add(SynE1I2)

    SynE2I1 = Synapses(E2, I, on_pre='v+=J_ppei')
    # connectivity type
    prelist, postlist = fixed_connectivity(N_i, K_ppei)
    SynE2I1.connect(i=prelist, j=postlist)
    # delays
    SynE2I1.delay = '{} * ms'.format(params['const_delay'])
    Net.add(SynE2I1)

    print('build synapses time: {}s'.format(time.time()-start_synapses))

    # initial distribution of the network simulation
    E.v = np.ones(len(E)) * params['net_delta_peak_E']*mV
    E2.v = np.ones(len(E2)) * params['net_delta_peak_E']*mV
    I.v = np.ones(len(I)) * params['net_delta_peak_I']*mV
    I2.v = np.ones(len(I2)) * params['net_delta_peak_I']*mV

    # initial distribution of w_mean
    if have_adap_e:
        # standard deviation of w_mean is set to 0.1
        E.w = 0.1 * np.random.randn(len(E)) * pA + net_w_init_e
        E2.w = 0.1 * np.random.randn(len(E2)) * pA + net_w_init_e
    if have_adap_i:
        # standard deviation of w_mean is set to 0.1
        I.w = 0.1 * np.random.randn(len(I)) * pA + net_w_init_i
        I2.w = 0.1 * np.random.randn(len(I2)) * pA + net_w_init_i

    # include a lower bound for the membrane voltage
    if 'net_v_lower_bound' in params and params['net_v_lower_bound'] is not None:
        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_E = E.run_regularly('v = clip(v, %s * mV, 10000 * mV)'
                                       % float(params['net_v_lower_bound']),
                                       when='end', order=-1, dt=dt_sim)
        print('Lower bound active at {}'.format(params['net_v_lower_bound']))
        Net.add(V_lowerbound_E)

        V_lowerbound_E2 = E2.run_regularly('v = clip(v, %s * mV, 10000 * mV)'
                                       % float(params['net_v_lower_bound']),
                                       when='end', order=-1, dt=dt_sim)
        print('Lower bound active at {}'.format(params['net_v_lower_bound']))
        Net.add(V_lowerbound_E2)


        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_I1 = I.run_regularly('v = clip(v, %s * mV, 10000 * mV)'
                                       % float(params['net_v_lower_bound']),
                                       when='end', order=-1, dt=dt_sim)
        print('Lower bound active at {}'.format(params['net_v_lower_bound']))
        Net.add(V_lowerbound_I1)

        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_I2 = I2.run_regularly('v = clip(v, %s * mV, 10000 * mV)'
                                       % float(params['net_v_lower_bound']),
                                       when='end', order=-1, dt=dt_sim)
        print('Lower bound active at {}'.format(params['net_v_lower_bound']))
        Net.add(V_lowerbound_I2)


    if record_all_v_at_times:
        # define clock which runs on a very course time grid (memory issue)
        clock_record_all = Clock(params['net_record_all_neurons_dt']*ms)
        v_monitor_record_all_E = StateMonitor(E, 'v', record=True, clock=clock_record_all)
        Net.add(v_monitor_record_all_E)
        v_monitor_record_all_E2 = StateMonitor(E2, 'v', record=True, clock=clock_record_all)
        Net.add(v_monitor_record_all_E2)
        v_monitor_record_all_I1 = StateMonitor(I, 'v', record=True, clock=clock_record_all)
        Net.add(v_monitor_record_all_I1)
        v_monitor_record_all_I2 = StateMonitor(I2, 'v', record=True, clock=clock_record_all)
        Net.add(v_monitor_record_all_I2)



    if record_spikes > 0:
        record_spikes_group_E = Subgroup(E, 0, min(record_spikes, N_e))
        spike_monitor_E = SpikeMonitor(record_spikes_group_E, name='E1_spikemon')
        Net.add(spike_monitor_E, record_spikes_group_E)

        record_spikes_group_E2 = Subgroup(E2, 0, min(record_spikes, N_e))
        spike_monitor_E2 = SpikeMonitor(record_spikes_group_E2, name='E2_spikemon')
        Net.add(spike_monitor_E2, record_spikes_group_E2)

        record_spikes_group_I1 = Subgroup(I, 0, min(record_spikes, N_i))
        spike_monitor_I1 = SpikeMonitor(record_spikes_group_I1, name='I1_spikemon')
        Net.add(spike_monitor_I1, record_spikes_group_I1)

        record_spikes_group_I2 = Subgroup(I2, 0, min(record_spikes, N_i))
        spike_monitor_I2 = SpikeMonitor(record_spikes_group_I2, name='I2_spikemon')
        Net.add(spike_monitor_I2, record_spikes_group_I2)



    print('------------------ running network!')
    start_time = time.time()
    Net.run(runtime, report='text')

    if params['brian2_standalone']:
        project_dir = cpp_default_dir + '/test' + str(os.getpid())
        device.build(directory=project_dir, compile=True, run=True)

    # extract results

    # unbinned quantities
    net_rates_e = rate_monitor_e.smooth_rate(window='flat', width=10.0*ms)/Hz
    net_t_e = rate_monitor_e.t/ms

    net_rates_e2 = rate_monitor_e2.smooth_rate(window='flat', width=10.0*ms)/Hz
    net_t_e2 = rate_monitor_e2.t/ms

    net_rates_i1 = rate_monitor_i.smooth_rate(window='flat', width=10.0*ms)/Hz
    net_t_i1 = rate_monitor_i.t/ms

    net_rates_i2 = rate_monitor_i2.smooth_rate(window='flat', width=10.0*ms)/Hz
    net_t_i2 = rate_monitor_i2.t/ms

    if record_spikes > 0:
        # multiply by 1 like this to ensure brian extracts the results before we delete the compile directory
        net_spikes_e = spike_monitor_E.it
        i, t = net_spikes_e
        i = i * 1; t = t * 1
        net_spikes_e = [i, t]

        net_spikes_e2 = spike_monitor_E2.it
        i, t = net_spikes_e2
        i = i * 1; t = t * 1
        net_spikes_e2 = [i, t]

        net_spikes_i1 = spike_monitor_I1.it
        i, t = net_spikes_i1
        i = i * 1; t = t * 1
        net_spikes_i1 = [i, t]

        net_spikes_i2 = spike_monitor_I2.it
        i, t = net_spikes_i2
        i = i * 1;
        t = t * 1
        net_spikes_i2 = [i, t]

    if record_all_v_at_times:
        v_all_neurons_e = v_monitor_record_all_E.v/mV
        t_all_neurons_e = v_monitor_record_all_E.t/ms
        v_all_neurons_e2 = v_monitor_record_all_E2.v/mV
        t_all_neurons_e2 = v_monitor_record_all_E2.t/ms
        v_all_neurons_i1 = v_monitor_record_all_I1.v/mV
        t_all_neurons_i1 = v_monitor_record_all_I1.t/ms
        v_all_neurons_i2 = v_monitor_record_all_I2.v/mV
        t_all_neurons_i2 = v_monitor_record_all_I2.t/ms

    run_time = time.time() - start_time
    print('runtime: %1.1f' % run_time)


    if params['brian2_standalone']:
        shutil.rmtree(project_dir)
        device.reinit()

    #for smoothing function net_rates do: helpers.smooth_trace(net_rates, int(rates_dt / dt_sim))
    # smooth out our hyper-resolution rate trace manually cause brian2 can't do it
    results_dict = {'brian_version': 2, 'r_e': net_rates_e, 'r_e2': net_rates_e2,
                    'r_i1': net_rates_i1,
                    'r_i2': net_rates_i2, 't': time_r}
    # results_dict = {'brian_version':2, 'r':net_rates, 't':net_t}
    # print(len(results_dict['t']))
    # time binning

    if record_spikes > 0:
        results_dict['net_spikes_e'] = net_spikes_e
        results_dict['net_spikes_e2'] = net_spikes_e2
        results_dict['net_spikes_i1'] = net_spikes_i1
        results_dict['net_spikes_i2'] = net_spikes_i2

    if record_all_v_at_times:
        results_dict['v_all_neurons_e'] = v_all_neurons_e
        results_dict['t_all_neurons_e'] = t_all_neurons_e
        results_dict['v_all_neurons_e2'] = v_all_neurons_e2
        results_dict['t_all_neurons_e2'] = t_all_neurons_e2
        results_dict['v_all_neurons_i1'] = v_all_neurons_i1
        results_dict['t_all_neurons_1i'] = t_all_neurons_i1
        results_dict['v_all_neurons_i2'] = v_all_neurons_i2
        results_dict['t_all_neurons_i2'] = t_all_neurons_i2
    return results_dict
