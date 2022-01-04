import numpy as np
from scipy.stats.stats import WeightedTauResult
import pyqmc.mc as mc
import scipy.stats
import pyqmc.linemin as linemin
"""
TODO:

 3) Parallel implementation of averager, with test

 4) Correlated sampling
    4.5) Correlated sampling test

 5) Optimizer
     5.5) Optimizer test
"""


def collect_overlap_data(wfs, configs, energy, transforms):
    r"""Collect the averages over configs assuming that
    configs are distributed according to

    .. math:: \rho \propto \sum_i |\Psi_i|^2

    The keys 'overlap' and 'overlap_gradient' are

    `overlap` :

    .. math:: \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    `overlap_gradient`:

    .. math:: \partial_m \langle \Psi_f | \Psi_i \rangle = \left\langle \frac{\partial_{fm} \Psi_i^* \Psi_j}{\rho} \right\rangle_{\rho}

    The function returns two dictionaries: 

    weighted_dat: each element is a list (one item per wf) of quantities that are accumulated as O psi_i^2/rho
    unweighted_dat: Each element is a numpy array that are accumulated just as O (no weight). 
                    This in particular includes 'weight' which is just psi_i^2/rho

    """
    phase, log_vals = [np.nan_to_num(np.array(x)) for x in zip(*[wf.value() for wf in wfs])]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    denominator = np.mean(np.exp(2 * (log_vals - ref)), axis=0)
    normalized_values = phase * np.exp(log_vals - ref)

    # Weight for quantities that are evaluated as
    # int( f(X) psi_f^2 dX )
    # since we sampled sum psi_i^2 /N 
    weight = np.exp(-2 * (log_vals[:, np.newaxis] - log_vals))
    weight = 1.0 / np.mean(weight, axis=1) # [wf, config]
    energies = [energy(configs, wf) for wf in wfs]

    dppsi = [transform.serialize_gradients(wf.pgradient()) for transform, wf in zip(transforms, wfs)] 

    weighted_dat = {}
    unweighted_dat  = {}

     # normalized_values are [config,wf]
     # we average over configs here and produce [wf,wf]
    unweighted_dat["overlap"] = np.einsum( 
        "ik,jk->ij", normalized_values.conj(), normalized_values / denominator
    ) / len(ref)

    #Weighted average
    for k in energies[0].keys():
        weighted_dat[k]=[]
    for wt, en in zip(weight, energies): 
        for k in en.keys():
            weighted_dat[k].append(np.mean(en[k]*wt,axis=0))

    weighted_dat['dpidpj'] = []
    weighted_dat['dppsi'] = []
    weighted_dat["dpH"] = []
    nconfig = weight.shape[1]
    for wfi, (dp,energy) in enumerate(zip(dppsi,energies)):
        weighted_dat['dppsi'].append(np.einsum(
            "ij,i->j", dp, weight[wfi] , optimize=True
        )/nconfig)
        weighted_dat['dpidpj'].append(np.einsum(
            "ij,i,ik->jk", dp, weight[wfi] , dp, optimize=True
        )/nconfig)
        weighted_dat["dpH"].append(np.einsum("i,ij,i->j", energy['total'], dp, weight[wfi])/nconfig)

    
    ## We have to be careful here because the wave functions may have different numbers of 
    ## parameters
    for wfi, dp in enumerate(dppsi):
        unweighted_dat[("overlap_gradient",wfi)] = \
            np.einsum(
                "km,ik,jk->ijm",  # shape (wf, param) k is config index
                dp,
                normalized_values.conj(),
                normalized_values / denominator,
            )/ len(ref)

    unweighted_dat["weight"] = np.mean(weight, axis=1)
    return weighted_dat, unweighted_dat


def invert_list_of_dicts(A):
    """
    if we have a list [ {'A':1,'B':2}, {'A':3, 'B':5}], invert the structure to 
    {'A':[1,3], 'B':[2,5]}. 
    If not all keys are present in all lists, error.
    """
    final_dict = {}
    for k in A[0].keys():
        final_dict[k] = []
    for a in A:
        for k, v in a.items():
            final_dict[k].append(v)
    return final_dict


def sample_overlap_worker(wfs, configs, energy, transforms, nsteps=10, nblocks=10, tstep=0.5):
    r"""Run nstep Metropolis steps to sample a distribution proportional to
    :math:`\sum_i |\Psi_i|^2`, where :math:`\Psi_i` = wfs[i]
    """
    nconf, nelec, _ = configs.configs.shape
    for wf in wfs:
        wf.recompute(configs)
    weighted = []
    unweighted=[]
    for block in range(nblocks):
        print('-', end="", flush=True)
        weighted_block = {}        
        unweighted_block = {}

        for n in range(nsteps):
            for e in range(nelec):  # a sweep
                # Propose move
                grads = [np.real(wf.gradient(e, configs.electron(e)).T) for wf in wfs]
                grad = mc.limdrift(np.mean(grads, axis=0))
                gauss = np.random.normal(scale=np.sqrt(tstep), size=(nconf, 3))
                newcoorde = configs.configs[:, e, :] + gauss + grad * tstep
                newcoorde = configs.make_irreducible(e, newcoorde)

                # Compute reverse move
                grads, vals = list(zip(*[wf.gradient_value(e, newcoorde) for wf in wfs]))
                grads = [np.real(g.T) for g in grads]
                new_grad = mc.limdrift(np.mean(grads, axis=0))
                forward = np.sum(gauss ** 2, axis=1)
                backward = np.sum((gauss + tstep * (grad + new_grad)) ** 2, axis=1)

                # Acceptance
                t_prob = np.exp(1 / (2 * tstep) * (forward - backward))
                wf_ratios = np.abs(vals) ** 2
                log_values = np.real(np.array([wf.value()[1] for wf in wfs]))
                weights = np.exp(2 * (log_values - log_values[0]))

                ratio = t_prob * np.sum(wf_ratios * weights, axis=0) / weights.sum(axis=0)
                accept = ratio > np.random.rand(nconf)
                #block_avg["acceptance"][n] += accept.mean() / nelec

                # Update wave function
                configs.move(e, newcoorde, accept)
                for wf in wfs:
                    wf.updateinternals(e, newcoorde, configs, mask=accept)

            # Collect rolling average
            weighted_dat, unweighted_dat = collect_overlap_data(wfs, configs, energy, transforms)
            for k, it in unweighted_dat.items():
                if k not in unweighted_block:
                    unweighted_block[k] = np.zeros((*it.shape,), dtype=it.dtype)
                unweighted_block[k] += unweighted_dat[k] / nsteps

            for k, it in weighted_dat.items():
                if k not in weighted_block:
                    weighted_block[k] = [np.zeros((*x.shape,), dtype=x.dtype) for x in it]
                for b, v in zip(weighted_block[k], it):
                    b += v / nsteps
        weighted.append(weighted_block)
        unweighted.append(unweighted_block)


    # here we modify the data so that it's a dictionary of lists of arrays for weighted
    # and a dictionary of arrays for unweighted
    # Access weighted as weighted[quantity][wave function][block, ...]
    # Access unweighted as unweighted[quantity][block,...]
    weighted = invert_list_of_dicts(weighted)
    unweighted = invert_list_of_dicts(unweighted)

    for k in weighted.keys():
        weighted[k] = [np.asarray(x) for x in map(list, zip(*weighted[k]))]
    for k in unweighted.keys():
        unweighted[k] = np.asarray(unweighted[k])
    print("sampling done")
    return weighted, unweighted, configs


def sample_overlap(wfs, configs, energy, transforms, nsteps=10, nblocks=10, tstep=0.5, client=None, npartitions=0):
    """
    """
    if client is None:
        return sample_overlap_worker(wfs, configs, energy, transforms, nsteps, nblocks, tstep)
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    runs = []
    for nodeconfigs in coord:
        runs.append(
            client.submit(
                sample_overlap_worker,
                wfs, nodeconfigs, energy, transforms, nsteps, nblocks, tstep
            )
        )
    allresults = list(zip(*[r.result() for r in runs]))
    configs.join(allresults[2])
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    #weighted = allresults[0]
    #unweighted=allresults[0]
    weighted = {}
    for k,it in invert_list_of_dicts(allresults[0]).items():
        inverted_array = [np.asarray(x) for x in map(list, zip(*it))] # make the wf index the outside one
        weighted[k] = [np.average(x, weights=confweight, axis=0) for x in inverted_array]
    unweighted = {}
    for k, it in invert_list_of_dicts(allresults[1]).items():
        unweighted[k] = np.average(np.asarray(it),weights=confweight, axis=0)

    return weighted, unweighted, configs


def average(weighted, unweighted):
    """
    (more or less) correctly average the output from sample_overlap
    Returns the average and error as dictionaries.

    TODO: use a more accurate formula for weighted uncertainties
    """
    avg = {}
    error = {}
    for k,it in weighted.items():
        avg[k] = []
        error[k] = []
        #weight is [block,wf], so we transpose
        for v, w in zip(it,unweighted['weight'].T): 
            avg[k].append(np.sum(v, axis=0)/np.sum(w))
            error[k].append(scipy.stats.sem(v,axis=0)/np.mean(w))
    for k,it in unweighted.items():
        avg[k] = np.mean(it, axis=0)
        error[k] = scipy.stats.sem(it, axis=0)
    return avg, error


def collect_terms(avg, error):
    """
    Generate the terms we need to do the optimization.
    """
    ret = {}

    nwf = len(avg['dpH'])
    ret['dp_energy'] = [2.0*np.real(dpH - total*dppsi) for dpH, total, dppsi in zip(avg['dpH'], avg['total'],avg['dppsi'])]
    ret['dp_norm'] = [2.0*np.real(avg[('overlap_gradient',i)][i,i,:]) for i in range(nwf)]
    ret['condition'] = [np.real(dpidpj - np.einsum("i,j->ij", dp, dp)) for dpidpj,dp in zip(avg['dpidpj'], avg['dppsi']) ]
    N = np.abs(avg["overlap"].diagonal())
    Nij = np.sqrt(np.outer(N, N))

    ret['norm'] = N
    ret['overlap'] = avg['overlap']/Nij
    # ends up being [i,j,m] where i, j are overlaps and m is the parameter
    ret['dp_overlap'] = [ (avg[('overlap_gradient',i)] -0.5 * avg['overlap'][:,:,np.newaxis]*ret['dp_norm'][i]/N[i] )/Nij[:,:,np.newaxis] for i in range(nwf)  ]
    return ret

def objective_function_derivative(terms, lam):
    """
    terms are output from generate_terms
    lam is the penalty
    """
    return  [dp_energy+
            lam * 2*np.sum(np.triu(np.rollaxis(dp_overlap,2)*terms['overlap'],1),axis=(1,2) ) +
            lam * 2*(N-1)*dp_norm
            for dp_energy, dp_overlap, N, dp_norm in zip(terms['dp_energy'], terms['dp_overlap'],terms['norm'], terms['dp_norm'])]




import pyqmc.hdftools as hdftools
import h5py
def hdf_save(hdf_file, data, attr):

    if hdf_file is not None:
        with h5py.File(hdf_file, "a") as hdf:
            if "energy" not in hdf.keys():
                hdftools.setup_hdf(hdf, data, attr)

            hdftools.append_hdf(hdf, data)


def correlated_sampling(wfs, configs, energy, transforms, parameters, client=None, npartitions=0):
    """
    Run in parallel if client is specified. 
    """
    if client is None:
        return correlated_sampling_worker(wfs, configs, energy, transforms, parameters)
    if npartitions is None:
        npartitions = sum(client.nthreads().values())

    coord = configs.split(npartitions)
    runs = []
    for nodeconfigs in coord:
        runs.append(
            client.submit(
                correlated_sampling_worker,
                wfs, nodeconfigs, energy, transforms, parameters
            )
        )
    allresults = [r.result() for r in runs]
    confweight = np.array([len(c.configs) for c in coord], dtype=float)
    weighted = {}
    for k,it in invert_list_of_dicts(allresults).items():
        weighted[k] = np.average(it, weights=confweight, axis=0) 

    return weighted




def correlated_sampling_worker(wfs, configs, energy, transforms, parameters):
    """
    Input: 
       wfs
       configs

    returns: 
      data along the path:
         overlap*weight_correlated
         energy*weight_correlated*weight_energy
         weights for correlated sampling: rhoprime /rho
         weights for energy expectation values: psi_i^2/rho
    """

    p0 = [transform.serialize_parameters(wf.parameters) for wf, transform in zip(wfs, transforms)]
    phase, log_vals = [np.nan_to_num(np.array(x)) for x in zip(*[wf.recompute(configs) for wf in wfs])]
    log_vals = np.real(log_vals)  # should already be real
    ref = np.max(log_vals, axis=0)
    weight_energy = np.exp(-2 * (log_vals[:, np.newaxis] - log_vals))
    weight_energy = 1.0 / np.mean(weight_energy, axis=1) # [wf, config]
    rho = np.sum(np.exp(2 * (log_vals - ref)), axis=0)
    nconfig = configs.configs.shape[0]

    weight_sample_final = []
    energy_final =[]
    overlap_final = []
    weight_variance_final = []
    for p, parameter in enumerate(parameters):
        for wf, transform, wf_parm in zip(wfs, transforms, parameter):
            for k, it in transform.deserialize(wf, wf_parm).items():
                wf.parameters[k] = it

        phase, log_vals = [np.nan_to_num(np.array(x)) for x in zip(*[wf.recompute(configs) for wf in wfs])]
        denominator = np.mean(np.exp(2 * (log_vals - ref)), axis=0)
        normalized_values = phase * np.exp(log_vals - ref)

        rhoprime = np.sum(np.exp(2 * (log_vals - ref)), axis=0)

        sampling_weight = rhoprime/rho
        # Weight for quantities that are evaluated as
        # int( f(X) psi_f^2 dX )
        # since we sampled sum psi_i^2 /N 
        weight = np.exp(-2 * (log_vals[:, np.newaxis] - log_vals))
        weight = 1.0 / np.mean(weight, axis=1) # [wf, config]
        energies = np.asarray([energy(configs, wf)['total'] for wf in wfs])
        overlap = np.einsum( 
            "ik,jk,k->ij", normalized_values.conj(), normalized_values / denominator,sampling_weight
        ) / len(ref)
        energies = np.einsum("ik, k, ik->i",energies, sampling_weight, weight_energy)/nconfig
        
        energy_final.append(energies)
        overlap_final.append(overlap)
        weight_variance_final.append(np.var(sampling_weight))
        weight_sample_final.append(np.mean(sampling_weight))
        #print('energy', energies/np.mean(sampling_weight*weight_energy))
        #print('overlap', overlap/np.mean(sampling_weight))

    for wf, transform, wf_parm in zip(wfs, transforms, p0):
        for k, it in transform.deserialize(wf, wf_parm).items():
            wf.parameters[k] = it
    return {'energy':np.asarray(energy_final),
            'overlap':np.asarray(overlap_final),
             'weight_sample':np.asarray(weight_sample_final),
             'weight_energy':np.mean(weight_energy, axis=1),
             'weight_variance':np.asarray(weight_variance_final)
    }

def find_move_from_line(x, data, penalty, weight_variance_threshold=0.1):
    """
    Given the data from correlated sampling, find the best move.

    Return: 
    cost function
    xmin estimation
    """
    energy = data['energy']/(data['weight_sample'][:,np.newaxis]*data['weight_energy'][np.newaxis,:])
    overlap = data['overlap']/data['weight_sample'][:,np.newaxis, np.newaxis]
    cost = np.sum(energy,axis=1) +\
           penalty*np.sum(np.triu(overlap**2,1),axis=(1,2)) +\
           penalty*np.einsum('ijj->i', (overlap-1)**2)
    good_points = data['weight_variance'] < weight_variance_threshold
    xmin = linemin.stable_fit(x[good_points],cost[good_points])
    return xmin, cost


        

def direct_move(grad, N=20, max_tstep=0.1):
        x = np.linspace(0,max_tstep, N)
        return [ [-delta*g for g in grad] for delta in x], x




def optimize(wfs, configs, energy, transforms, hdf_file, penalty=.5, nsteps=40, max_tstep=0.1, 
            diagonal_approximation=False,
            condition_epsilon=0.1,
            client=None,
            npartitions=0):
    parameters = [transform.serialize_parameters(wf.parameters) 
          for transform, wf in zip(transforms, wfs)]
    
    data = {}
    for k in ['energy','parameters','norm','overlap', 'energy_error']:
        data[k] = []
    for step in range(nsteps):
        data_weighted, data_unweighted, configs = sample_overlap(wfs,configs, energy, transforms, nsteps=10, nblocks=40, client=client, npartitions=npartitions)
        avg, error = average(data_weighted, data_unweighted)
        print('energy', avg['total'], error['total'])
        terms = collect_terms(avg,error)
        print('norm',terms['norm'])
        print('overlap', terms['overlap'][0,1])
        derivative = objective_function_derivative(terms,penalty)
        if diagonal_approximation:
            derivative_conditioned = [d/(condition.diagonal()+condition_epsilon) for d, condition in zip(derivative,terms['condition'])]
        else:
            derivative_conditioned = [ -linemin.sr_update(d,condition,1.0) for d,condition in zip(derivative,terms['condition'])]
        # 
        print('|gradient|',[np.linalg.norm(d) for d in derivative_conditioned])

        line_parameters,x = direct_move(derivative_conditioned, max_tstep=max_tstep)
        for line_p in line_parameters:
            for p, p0 in zip(line_p, parameters):
                p+=p0
            
        correlated_data = correlated_sampling(wfs, configs, energy, transforms, line_parameters, client=client, npartitions=npartitions)
        xmin, cost = find_move_from_line(x,correlated_data, penalty)
        print('line search', x,cost)
        print("choosing to move", xmin)
        if abs(xmin) < 1e-16: # if we didn't move at all
            max_tstep=0.5*max_tstep
        else:
            max_tstep=2*xmin
        print("setting the step range to ", max_tstep)
        parameters = [p - xmin*d for p,d in zip(parameters, derivative_conditioned)]
        for wf, transform, parm in zip(wfs, transforms, parameters):
            for k, it in transform.deserialize(wf, parm).items():
                wf.parameters[k] = it


class AdamMove():
    def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon

    def update(self, g,m_old,v_old,t):
        m_new = self.beta1*m_old + (1-self.beta1)*g
        v_new = self.beta2*v_old + (1-self.beta2)*g**2
        mhat = m_new/(1-self.beta1**t)
        vhat = v_new/(1-self.beta2**t)
        theta_move = -self.alpha*mhat/(np.sqrt(vhat)+self.epsilon)
        return theta_move, m_new, v_new


def optimize_adam(wfs, configs, energy, transforms, hdf_file, penalty=.5, nsteps=400, alpha=0.01, beta1=0.9):
    adam = AdamMove(alpha=alpha, beta1=beta1)
    parameters = [transform.serialize_parameters(wf.parameters) 
          for transform, wf in zip(transforms, wfs)]
    m_adam = [np.zeros_like(x) for x in parameters]
    v_adam = [np.zeros_like(x) for x in parameters]
    data = {}
    for k in ['energy','parameters','norm','overlap', 'energy_error']:
        data[k] = []
    for step in range(nsteps):
        data_weighted, data_unweighted, configs = sample_overlap_worker(wfs,configs, energy, transforms, nsteps=10, nblocks=40)
        avg, error = average(data_weighted, data_unweighted)
        print('energy', avg['total'], error['total'])
        terms = collect_terms(avg,error)
        print('norm',terms['norm'])
        print('overlap', terms['overlap'][0,1])
        derivative = objective_function_derivative(terms,penalty)
        derivative_conditioned = [d/np.sqrt(condition.diagonal()) for d, condition in zip(derivative,terms['condition'])]
        print('|gradient|',[np.linalg.norm(d) for d in derivative_conditioned])

        adam_moves = [adam.update(g,m, v, step+1) for g,m,v in zip(derivative_conditioned, m_adam, v_adam)]
        m_adam = [move[1] for move in adam_moves]
        v_adam = [move[2] for move in adam_moves]
        parameters = [parm+move[0] for parm, move in zip(parameters,adam_moves)]
        print('parameters', [param.real.round(3) for param in parameters])
        for wf, transform, parm in zip(wfs, transforms, parameters):
            for k, it in transform.deserialize(wf, parm).items():
                wf.parameters[k] = it

        data = {'energy':avg['total'],
                  'energy_error': error['total'],
                  'norm':terms['norm'],
                  'overlap':terms['overlap'][0,1]
                  }
        for i,parm in enumerate(parameters):
            data[f'parameters_{i}'] = parm

        hdf_save(hdf_file, 
                  data,
                  {'alpha':alpha,
                  'beta1':beta1,
                   'nconfig':configs.configs.shape[0],
                   'penalty':penalty})