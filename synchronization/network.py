from __future__ import print_function

from brian2 import *

import os
import time
import shutil
import gc

cpp_default_dir = "brian2_compile"


def network_sim(signal, params: dict):
    if params["brian2_standalone"]:
        set_device(params["brian2_device"], build_on_run=False)
        device.insert_code(
            "main", "srand(" + str(int(time.time()) + os.getpid()) + ");"
        )
    else:
        prefs.codegen.target = params["brian2_codegen_target"]

    # stripe on brian units within the copied dict for the simulation so that brian can work with them

    ### Neuron group specific parameters
    ### Excitatory cells
    C_e = params["C_exc"] * pF
    gL_e = params["gL_exc"] * nS
    R_e = 1 / gL_e
    taum_e = params["taum_exc"] * ms
    EL_e = params["EL_exc"] * mV
    Ew_e = params["Ew_exc"] * mV
    VT_e = params["VT_exc"] * mV
    negVT_e = -VT_e
    deltaT_e = params["deltaT_exc"] * mV
    Vcut_e = params["Vcut_exc"] * mV
    tauw_e = params["tauw_exc"] * ms
    Vr_e = params["Vr_exc"] * mV
    t_ref_e = params["t_ref_exc"] * ms
    tau_AMPA = params["tau_AMPA"] * ms
    E_AMPA = params["E_AMPA"] * mV

    ### Inhibitory cells
    C_i = params["C_inh1"] * pF
    gL_i = params["gL_inh1"] * nS
    R_i = 1 / gL_i
    taum_i = params["taum_inh1"] * ms
    EL_i = params["EL_inh1"] * mV
    Ew_i = params["Ew_inh1"] * mV
    VT_i = params["VT_inh1"] * mV
    negVT_i = -VT_i
    deltaT_i = params["deltaT_inh1"] * mV
    Vcut_i = params["Vcut_inh1"] * mV
    tauw_i = params["tauw_inh1"] * ms
    Vr_i = params["Vr_inh1"] * mV
    t_ref_i = params["t_ref_inh1"] * ms
    tau_GABA = params["tau_GABA"] * ms
    E_GABA = params["E_GABA"] * mV

    ### General parameters
    dt_sim = params["net_dt"] * ms
    net_w_init_e = params["net_w_init_e"] * pA
    net_w_init_i = params["net_w_init_i"] * pA
    rates_dt = params["net_record_dt"] * ms
    runtime = params["runtime"] * ms
    time_r = np.arange(0.0, runtime / ms, rates_dt / ms)
    time_v = np.arange(0.0, runtime / ms, dt_sim / ms)

    # ratio used for smoothing the spike histogram to match resolution
    ratesdt_dtsim = int(rates_dt / dt_sim)
    N_pop = params["N_pop"]
    N_e = params["N_e"]
    N_i = params["N_i"]

    # garbage collect so we can run multiple brian2 runs
    gc.collect()

    # seed our random number generator!  necessary on the unix server
    np.random.seed()

    mu_ext_array_e = signal[0]  # [mV/ms]
    sigma_ext_array_e = signal[1]  # [mV/sqrt(ms)]

    # todo: Change runner.py to create specific input for inhibitory population(s)
    mu_ext_array_i = np.zeros_like(mu_ext_array_e)  # signal[0] # [mV/ms]
    sigma_ext_array_i = np.zeros_like(mu_ext_array_e)  # signal[1] # [mV/sqrt(ms)]

    # what is recorded during the simulation
    record_spikes = params["net_record_spikes"]
    record_all_v_at_times = params["net_record_all_neurons"]

    # simulation time step
    simclock = Clock(dt_sim)

    w_refr_e = (
        " (unless refractory)"
        if "net_w_refr_e" not in params or params["net_w_refr_e"]
        else ""
    )

    w_refr_i = (
        " (unless refractory)"
        if "net_w_refr_i" not in params or params["net_w_refr_i"]
        else ""
    )

    a_e = params["a_exc"]
    b_e = params["b_exc"]
    # convert to array if adapt params are scalar values
    if type(a_e) in [int, float]:
        a_e = np.ones_like(mu_ext_array_e) * a_e
    if type(b_e) in [int, float]:
        b_e = np.ones_like(mu_ext_array_e) * b_e

    a_i = params["a_inh1"]
    b_i = params["b_inh1"]
    # convert to array if adapt params are scalar values
    if type(a_i) in [int, float]:
        a_i = np.ones_like(mu_ext_array_e) * a_i
    if type(b_i) in [int, float]:
        b_i = np.ones_like(mu_ext_array_e) * b_i

    # decide if there's adaptation
    have_adap_e = True if (a_e.any() > 0.0) or (b_e.any() > 0.0) else False
    have_adap_i = True if (a_i.any() > 0.0) or (b_i.any() > 0.0) else False

    # convert numpy arrays to TimedArrays
    a_e = TimedArray(a_e * nS, dt_sim)
    b_e = TimedArray(b_e * pA, dt_sim)
    a_i = TimedArray(a_i * nS, dt_sim)
    b_i = TimedArray(b_i * pA, dt_sim)

    # transform the external input into TimedArray
    mu_ext_e = TimedArray(mu_ext_array_e * (mV / ms), dt_sim)
    sigma_ext_e = TimedArray(sigma_ext_array_e * (mV / sqrt(ms)), dt_sim)
    mu_ext_i = TimedArray(mu_ext_array_i * (mV / ms), dt_sim)
    sigma_ext_i = TimedArray(sigma_ext_array_i * (mV / sqrt(ms)), dt_sim)

    model_term_e = "((EL_e - v) + deltaT_e * exp((negVT_e + v) / deltaT_e)) / taum_e"
    model_term_i = "((EL_i - v) + deltaT_i * exp((negVT_i + v) / deltaT_i)) / taum_i"

    # TODO: use mu_ext_e[i](t) to access noise by neuron index, generate N * noise signals beforehand and add to array
    # (https://brian2.readthedocs.io/en/stable/user/input.html)

    # TODO: use named subexpressions instead of string formatting
    model_eqs_e1 = f"""
        dv/dt = %s %s {"+ mu_ext_e(t) + sigma_ext_e(t) * xi" if params["ou_enabled"][0] else ""} + I_syn_AMPA/C_e + I_syn_GABA/C_e : volt (unless refractory)
        %s
        I_syn_AMPA = g_ampa*(E_AMPA-v): amp # synaptic current
        dg_ampa/dt = -g_ampa/tau_AMPA : siemens # synaptic conductance
        I_syn_GABA = g_gaba*(E_GABA-v): amp # synaptic current
        dg_gaba/dt = -g_gaba/tau_GABA : siemens # synaptic conductance
        """ % (
        model_term_e,
        "- (w / C_e)" if have_adap_e else "",
        ("dw/dt = (a_e(t) * (v - Ew_e) - w) / tauw_e : amp %s" % w_refr_e)
        if have_adap_e
        else "",
    )

    model_eqs_i1 = f"""
        dv/dt = %s %s {"+ mu_ext_i(t) + sigma_ext_i(t) * xi" if params["ou_enabled"][0] else ""} + I_syn_AMPA/C_e + I_syn_GABA/C_e : volt (unless refractory)
        %s
        I_syn_AMPA = g_ampa*(E_AMPA-v): amp # synaptic current
        dg_ampa/dt = -g_ampa/tau_AMPA : siemens # synaptic conductance
        I_syn_GABA = g_gaba*(E_GABA-v): amp # synaptic current
        dg_gaba/dt = -g_gaba/tau_GABA : siemens # synaptic conductance
        """ % (
        model_term_i,
        "- (w / C_e)" if have_adap_e else "",
        ("dw/dt = (a_e(t) * (v - Ew_e) - w) / tauw_e : amp %s" % w_refr_i)
        if have_adap_e
        else "",
    )

    # Synapse Parameters
    # These parameters have to stay in global scope for Brian2 even if they are unused right now
    J_etoe = params["J_etoe"] * nS
    J_etoi = params["J_etoi"] * nS
    J_itoe = params["J_itoe"] * nS
    J_itoi = params["J_itoi"] * nS
    J_ppee = params["J_ppee"] * nS
    J_ppei = params["J_ppei"] * nS
    p_etoe = params["p_etoe"]
    p_etoi = params["p_etoi"]
    p_itoe = params["p_itoe"]
    p_itoi = params["p_itoi"]
    p_ppee = params["p_ppee"]
    p_ppei = params["p_ppei"]

    E, I = create_neuron_group_1(
        N_e,
        N_i,
        have_adap_e,
        have_adap_i,
        model_eqs_e1,
        model_eqs_i1,
        params,
        simclock,
        t_ref_e,
        t_ref_i,
    )

    if N_pop > 1:
        mu_ext_array_i2 = np.zeros_like(mu_ext_array_e)  # signal[0] # [mV/ms]
        sigma_ext_array_i2 = np.zeros_like(mu_ext_array_e)  # signal[1] # [mV/sqrt(ms)]

        mu_ext_array_e2 = signal[2]  # [mV/ms]
        sigma_ext_array_e2 = signal[3]  # [mV/sqrt(ms)]

        mu_ext_e2 = TimedArray(mu_ext_array_e2 * (mV / ms), dt_sim)
        sigma_ext_e2 = TimedArray(sigma_ext_array_e2 * (mV / sqrt(ms)), dt_sim)
        mu_ext_i2 = TimedArray(mu_ext_array_i2 * (mV / ms), dt_sim)
        sigma_ext_i2 = TimedArray(sigma_ext_array_i2 * (mV / sqrt(ms)), dt_sim)

        noise_e2 = "+ mu_ext_e2(t) + sigma_ext_e2(t) * xi"

        model_eqs_e2 = f"""
        dv/dt = %s %s {noise_e2 if params["ou_enabled"][1] else ""} + I_syn_AMPA/C_e + I_syn_GABA/C_e : volt (unless refractory)
        %s
        I_syn_AMPA = g_ampa*(E_AMPA-v): amp # synaptic current
        dg_ampa/dt = -g_ampa/tau_AMPA : siemens # synaptic conductance
        I_syn_GABA = g_gaba*(E_GABA-v): amp # synaptic current
        dg_gaba/dt = -g_gaba/tau_GABA : siemens # synaptic conductance
        """ % (
            model_term_e,
            "- (w / C_e)" if have_adap_e else "",
            ("dw/dt = (a_e(t) * (v - Ew_e) - w) / tauw_e : amp %s" % w_refr_e)
            if have_adap_e
            else "",
        )

        model_eqs_i2 = f"""
        dv/dt = %s %s {"+ mu_ext_i2(t) + sigma_ext_i2(t) * xi" if params["ou_enabled"][1] else ""} + I_syn_AMPA/C_e + I_syn_GABA/C_e : volt (unless refractory)
        %s
        I_syn_AMPA = g_ampa*(E_AMPA-v): amp # synaptic current
        dg_ampa/dt = -g_ampa/tau_AMPA : siemens # synaptic conductance
        I_syn_GABA = g_gaba*(E_GABA-v): amp # synaptic current
        dg_gaba/dt = -g_gaba/tau_GABA : siemens # synaptic conductance
        """ % (
            model_term_i,
            "- (w / C_e)" if have_adap_e else "",
            ("dw/dt = (a_e(t) * (v - Ew_e) - w) / tauw_e : amp %s" % w_refr_e)
            if have_adap_e
            else "",
        )

        E2, I2 = create_neuron_group_2(
            N_e,
            N_i,
            have_adap_e,
            have_adap_i,
            model_eqs_e2,
            model_eqs_i2,
            params,
            simclock,
            t_ref_e,
            t_ref_i,
        )

    rate_monitor_e = PopulationRateMonitor(E, name="aeif_ratemon_e")
    rate_monitor_i = PopulationRateMonitor(I, name="aeif_ratemon_i")

    print("Initializing net ...")
    start_init = time.time()

    net = Network(E, I, rate_monitor_e, rate_monitor_i,)

    if params["poisson_enabled"][0]:
        # dv_1 = sigma^2 / mu
        poisson_strength_1 = params["poisson_variance"] / params["poisson_mean_input"]

        # lambda_1 = mu / dv_1 / #neurons
        group_rate_ = params["poisson_mean_input"] / poisson_strength_1
        neuron_rate_ = group_rate_ / params["poisson_size"]
        print(f"Net 1 - poisson rate {group_rate_} - single neuron {neuron_rate_}")
        print(f"Poisson strength: {poisson_strength_1}")

        P_E = PoissonInput(
            target=E,
            target_var="v",
            N=params["poisson_size"],
            rate=neuron_rate_ * Hz,
            weight="((VT_e - Vr_e) * poisson_strength_1 * taum_e) / R_e / C_e",
        )
        net.add(P_E)

        if params["poisson_enabled_I"]:
            P_I = PoissonInput(
                target=I,
                target_var="v",
                N=params["poisson_size"],
                rate=neuron_rate_ * Hz,
                weight="((VT_i - Vr_i) * poisson_strength_1 * taum_i) / R_i / C_i",
            )
            net.add(P_I)

    build_synapses_first_population(
        E, I, p_etoe, p_etoi, p_itoe, p_itoi, N_e, N_i, net, simclock
    )

    if N_pop > 1:
        rate_monitor_i2 = PopulationRateMonitor(I2, name="aeif_ratemon_i2")
        rate_monitor_e2 = PopulationRateMonitor(E2, name="aeif_ratemon_e2")
        net.add(E2, I2, rate_monitor_e2, rate_monitor_i2)

        if params["poisson_enabled"][1]:
            # dv_2 = p * dv_1
            poisson_strength_1 = (
                params["poisson_variance"] / params["poisson_mean_input"]
            )

            # second network gets higher strength which leads to lower rate.
            # dv_2
            # poisson_strength_2 = poisson_strength_1 * params["poisson_p"]

            # lambda_2 = mu / dv_2 / #neurons
            # rate_2 = (
            #     params["poisson_mean_input"]
            #     / poisson_strength_2
            #     / params["poisson_size"]
            # )

            # rate of 2nd network is fraction of 1st network.
            rate_2 = neuron_rate_ * params["poisson_p"]

            print(f"Net 2 - rate for single neuron {rate_2}")

            P_E_2 = PoissonInput(
                target=E2,
                target_var="v",
                N=params["poisson_size"],
                rate=rate_2 * Hz,
                weight="((VT_e - Vr_e) * poisson_strength_1 * taum_e) / R_e / C_e",
            )
            net.add(P_E_2)

            if params["poisson_enabled_I"]:
                P_I_2 = PoissonInput(
                    target=I2,
                    target_var="v",
                    N=params["poisson_size"],
                    rate=rate_2 * Hz,
                    weight="((VT_i - Vr_i) * poisson_strength_1 * taum_i) / R_i / C_i",
                )
                net.add(P_I_2)

        build_synapses_multiple_populations(
            E,
            E2,
            I,
            I2,
            p_etoe,
            p_etoi,
            p_itoe,
            p_itoi,
            p_ppee,
            p_ppei,
            N_e,
            N_i,
            net,
            params,
            simclock,
        )

    print("Initialization time: {}s".format(time.time() - start_init))

    # Initial distribution of the network simulation
    if params["net_random_membrane_voltage"]:
        # Borrowed from https://brian2.readthedocs.io/en/stable/examples/frompapers.Stimberg_et_al_2018.example_1_COBA.html
        E.v = "EL_e + rand() * (VT_e - EL_e)"
        I.v = "EL_i + rand() * (VT_i - EL_i)"
    else:
        E.v = np.ones(len(E)) * params["net_delta_peak_E"] * mV
        I.v = np.ones(len(I)) * params["net_delta_peak_I"] * mV

    if N_pop > 1:
        if params["net_random_membrane_voltage"]:
            E2.v = "EL_e + rand() * (VT_e - EL_e)"
            I2.v = "EL_i + rand() * (VT_i - EL_i)"
        else:
            E2.v = np.ones(len(E2)) * params["net_delta_peak_E"] * mV
            I2.v = np.ones(len(I2)) * params["net_delta_peak_I"] * mV

    # Initial distribution of w_mean
    if have_adap_e:
        # standard deviation of w_mean is set to 0.1
        E.w = 0.1 * np.random.randn(len(E)) * pA + net_w_init_e

        if N_pop > 1:
            E2.w = 0.1 * np.random.randn(len(E2)) * pA + net_w_init_e

    if have_adap_i:
        # standard deviation of w_mean is set to 0.1
        I.w = 0.1 * np.random.randn(len(I)) * pA + net_w_init_i

        if N_pop > 1:
            I2.w = 0.1 * np.random.randn(len(I2)) * pA + net_w_init_i

    # include a lower bound for the membrane voltage
    if "net_v_lower_bound" in params and params["net_v_lower_bound"] is not None:
        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_E = E.run_regularly(
            "v = clip(v, %s * mV, 10000 * mV)" % float(params["net_v_lower_bound"]),
            when="end",
            order=-1,
            dt=dt_sim,
        )

        print("Lower bound active at {}".format(params["net_v_lower_bound"]))
        net.add(V_lowerbound_E)

        V_lowerbound_E2 = E2.run_regularly(
            "v = clip(v, %s * mV, 10000 * mV)" % float(params["net_v_lower_bound"]),
            when="end",
            order=-1,
            dt=dt_sim,
        )

        print("Lower bound active at {}".format(params["net_v_lower_bound"]))
        net.add(V_lowerbound_E2)

        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_I1 = I.run_regularly(
            "v = clip(v, %s * mV, 10000 * mV)" % float(params["net_v_lower_bound"]),
            when="end",
            order=-1,
            dt=dt_sim,
        )

        print("Lower bound active at {}".format(params["net_v_lower_bound"]))
        net.add(V_lowerbound_I1)

        # new in Brian2.0b4: custom_operation --> run_regularly
        V_lowerbound_I2 = I2.run_regularly(
            "v = clip(v, %s * mV, 10000 * mV)" % float(params["net_v_lower_bound"]),
            when="end",
            order=-1,
            dt=dt_sim,
        )

        print("Lower bound active at {}".format(params["net_v_lower_bound"]))
        net.add(V_lowerbound_I2)

    if record_all_v_at_times:
        # define clock which runs on a very course time grid (memory issue)
        clock_record_all = Clock(params["net_record_all_neurons_dt"] * ms)
        v_monitor_record_all_E = StateMonitor(
            E, "v", record=True, clock=clock_record_all
        )
        net.add(v_monitor_record_all_E)

        v_monitor_record_all_I1 = StateMonitor(
            I, "v", record=True, clock=clock_record_all
        )
        net.add(v_monitor_record_all_I1)

        if N_pop > 1:
            v_monitor_record_all_E2 = StateMonitor(
                E2, "v", record=True, clock=clock_record_all
            )
            net.add(v_monitor_record_all_E2)

            v_monitor_record_all_I2 = StateMonitor(
                I2, "v", record=True, clock=clock_record_all
            )
            net.add(v_monitor_record_all_I2)

    if record_spikes > 0:
        record_spikes_group_E = Subgroup(E, 0, min(record_spikes, N_e))
        spike_monitor_E = SpikeMonitor(record_spikes_group_E, name="E1_spikemon")
        net.add(spike_monitor_E, record_spikes_group_E)

        record_spikes_group_I1 = Subgroup(I, 0, min(record_spikes, N_i))
        spike_monitor_I1 = SpikeMonitor(record_spikes_group_I1, name="I1_spikemon")
        net.add(spike_monitor_I1, record_spikes_group_I1)

        if N_pop > 1:
            record_spikes_group_E2 = Subgroup(E2, 0, min(record_spikes, N_e))
            spike_monitor_E2 = SpikeMonitor(record_spikes_group_E2, name="E2_spikemon")
            net.add(spike_monitor_E2, record_spikes_group_E2)

            record_spikes_group_I2 = Subgroup(I2, 0, min(record_spikes, N_i))
            spike_monitor_I2 = SpikeMonitor(record_spikes_group_I2, name="I2_spikemon")
            net.add(spike_monitor_I2, record_spikes_group_I2)

    print("==== Running Network ... ====")
    start_time = time.time()
    net.run(runtime, report="text")
    print("==== Network Run Finished ====")

    if params["brian2_standalone"]:
        project_dir = cpp_default_dir + "/test" + str(os.getpid())
        device.build(directory=project_dir, compile=True, run=True, debug=False)

    run_time = time.time() - start_time
    print("runtime: %1.1f" % run_time)

    # unbinned quantities
    net_rates_e = rate_monitor_e.smooth_rate(window="flat", width=10.0 * ms) / Hz
    net_t_e = rate_monitor_e.t / ms

    net_rates_i1 = rate_monitor_i.smooth_rate(window="flat", width=10.0 * ms) / Hz
    net_t_i1 = rate_monitor_i.t / ms

    # for smoothing function net_rates do: helpers.smooth_trace(net_rates, int(rates_dt / dt_sim))
    # smooth out our hyper-resolution rate trace manually cause brian2 can't do it
    results = {
        "brian_version": 2,
        "r_e": net_rates_e,
        "r_e_t": net_t_e,
        "r_i1": net_rates_i1,
        "r_i1_t": net_t_i1,
        "t": time_r,
        "dt_sim": dt_sim,
    }

    if N_pop > 1:
        net_rates_e2 = rate_monitor_e2.smooth_rate(window="flat", width=10.0 * ms) / Hz
        net_t_e2 = rate_monitor_e2.t / ms

        net_rates_i2 = rate_monitor_i2.smooth_rate(window="flat", width=10.0 * ms) / Hz
        net_t_i2 = rate_monitor_i2.t / ms

        results["r_e2"] = net_rates_e2
        results["r_e2_t"] = net_t_e2
        results["r_i2"] = net_rates_i2
        results["r_i2_t"] = net_t_i2

    if record_spikes > 0:
        # multiply by 1 like this to ensure brian extracts the results before we delete the compile directory
        net_spikes_e = spike_monitor_E.it
        i, t = net_spikes_e
        net_spikes_e = [i * 1, t * 1]

        net_spikes_i1 = spike_monitor_I1.it
        i, t = net_spikes_i1
        net_spikes_i1 = [i * 1, t * 1]

        results["net_spikes_e"] = net_spikes_e
        results["net_spikes_i1"] = net_spikes_i1

        if N_pop > 1:
            net_spikes_e2 = spike_monitor_E2.it
            i, t = net_spikes_e2
            net_spikes_e2 = [i * 1, t * 1]

            net_spikes_i2 = spike_monitor_I2.it
            i, t = net_spikes_i2
            net_spikes_i2 = [i * 1, t * 1]

            results["net_spikes_e2"] = net_spikes_e2
            results["net_spikes_i2"] = net_spikes_i2

    if record_all_v_at_times:
        v_all_neurons_e = v_monitor_record_all_E.v / mV
        t_all_neurons_e = v_monitor_record_all_E.t / ms

        v_all_neurons_i1 = v_monitor_record_all_I1.v / mV
        t_all_neurons_i1 = v_monitor_record_all_I1.t / ms

        results["v_all_neurons_e"] = v_all_neurons_e
        results["t_all_neurons_e"] = t_all_neurons_e

        results["v_all_neurons_i1"] = v_all_neurons_i1
        results["t_all_neurons_i1"] = t_all_neurons_i1

        if N_pop > 1:
            v_all_neurons_e2 = v_monitor_record_all_E2.v / mV
            t_all_neurons_e2 = v_monitor_record_all_E2.t / ms

            v_all_neurons_i2 = v_monitor_record_all_I2.v / mV
            t_all_neurons_i2 = v_monitor_record_all_I2.t / ms

            results["v_all_neurons_e2"] = v_all_neurons_e2
            results["t_all_neurons_e2"] = t_all_neurons_e2

            results["v_all_neurons_i2"] = v_all_neurons_i2
            results["t_all_neurons_i2"] = t_all_neurons_i2

    if params["brian2_standalone"]:
        shutil.rmtree(project_dir)
        device.reinit()

    return results


def build_synapses_multiple_populations(
    E,
    E2,
    I,
    I2,
    p_etoe,
    p_etoi,
    p_itoe,
    p_itoi,
    p_ppee,
    p_ppei,
    N_e,
    N_i,
    net,
    params,
    simclock,
):
    syn_EE2 = Synapses(E2, E2, on_pre="g_ampa+=J_etoe", clock=simclock)
    syn_EE2.connect(p=p_etoe)
    net.add(syn_EE2)

    syn_EI2 = Synapses(E2, I2, on_pre="g_ampa+=J_etoi", clock=simclock)
    syn_EI2.connect(p=p_etoi)
    net.add(syn_EI2)

    syn_IE2 = Synapses(I2, E2, on_pre="g_gaba+=J_itoe", clock=simclock)
    syn_IE2.connect(p=p_itoe)
    net.add(syn_IE2)

    synII2 = Synapses(I2, I2, on_pre="g_gaba+=J_itoi", clock=simclock)
    synII2.connect(p=p_itoi)
    net.add(synII2)

    """
    Connections between populations
    """
    SynE1E2 = Synapses(E, E2, on_pre="g_ampa+=J_ppee", clock=simclock)
    SynE1E2.connect(p=p_ppee)
    SynE1E2.delay = "{} * ms".format(params["const_delay"], clock=simclock)
    net.add(SynE1E2)

    SynE2E1 = Synapses(E2, E, on_pre="g_ampa+=J_ppee", clock=simclock)
    SynE2E1.connect(p=p_ppee)
    SynE2E1.delay = "{} * ms".format(params["const_delay"], clock=simclock)
    net.add(SynE2E1)

    SynE1I2 = Synapses(E, I2, on_pre="g_ampa+=J_ppei", clock=simclock)
    SynE1I2.connect(p=p_ppei)
    SynE1I2.delay = "{} * ms".format(params["const_delay"])
    net.add(SynE1I2)

    SynE2I1 = Synapses(E2, I, on_pre="g_ampa+=J_ppei", clock=simclock)
    SynE2I1.connect(p=p_ppei)
    SynE2I1.delay = "{} * ms".format(params["const_delay"])
    net.add(SynE2I1)


def build_synapses_first_population(
    E, I, p_etoe, p_etoi, p_itoe, p_itoi, N_e, N_i, net, simclock
):
    synEE = Synapses(E, E, on_pre="g_ampa+=J_etoe", clock=simclock)
    synEE.connect(p=p_etoe)
    net.add(synEE)

    synEI = Synapses(E, I, on_pre="g_ampa+=J_etoi", clock=simclock)
    synEI.connect(p=p_etoi)
    net.add(synEI)

    syn_IE = Synapses(I, E, on_pre="g_gaba+=J_itoe", clock=simclock)
    syn_IE.connect(p=p_itoe)
    net.add(syn_IE)

    syn_II = Synapses(I, I, on_pre="g_gaba+=J_itoi", clock=simclock)
    syn_II.connect(p=p_itoi)
    net.add(syn_II)


def create_neuron_group_2(
    N_e,
    N_i,
    have_adap_e,
    have_adap_i,
    model_eqs_e2,
    model_eqs_i2,
    params,
    simclock,
    t_ref_e,
    t_ref_i,
):
    E2 = NeuronGroup(
        N=N_e,
        model=model_eqs_e2,
        threshold="v > Vcut_e",
        clock=simclock,
        reset="v = Vr_e%s" % ("; w += b_e(t)" if have_adap_e else ""),
        refractory=t_ref_e,
        method=params["net_integration_method"],
    )

    I2 = NeuronGroup(
        N=N_i,
        model=model_eqs_i2,
        threshold="v > Vcut_i",
        clock=simclock,
        reset="v = Vr_i%s" % ("; w += b_i(t)" if have_adap_i else ""),
        refractory=t_ref_i,
        method=params["net_integration_method"],
    )
    return E2, I2


def create_neuron_group_1(
    N_e,
    N_i,
    have_adap_e,
    have_adap_i,
    model_eqs_e1,
    model_eqs_i1,
    params,
    simclock,
    t_ref_e,
    t_ref_i,
):
    E = NeuronGroup(
        N=N_e,
        model=model_eqs_e1,
        threshold="v > Vcut_e",
        clock=simclock,
        reset="v = Vr_e%s" % ("; w += b_e(t)" if have_adap_e else ""),
        refractory=t_ref_e,
        method=params["net_integration_method"],
    )

    I = NeuronGroup(
        N=N_i,
        model=model_eqs_i1,
        threshold="v > Vcut_i",
        clock=simclock,
        reset="v = Vr_i%s" % ("; w += b_i(t)" if have_adap_i else ""),
        refractory=t_ref_i,
        method=params["net_integration_method"],
    )
    return E, I
