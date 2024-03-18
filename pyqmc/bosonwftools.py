import bosonslater as slater
#import pyqmc.slater as slater
# import bosonmultiplywf as multiplywf
import jastrowspin
import multiplywf
import pyqmc.gpu as gpu
import numpy as np

from wftools import default_jastrow_basis, read_wf, generate_jastrow


def generate_boson(
    mol,
    mf,
    mc=None,
    optimize_determinants=False,
    optimize_orbitals=False,
    optimize_zeros=True,    
    epsilon=1e-8,
    **kwargs,
):
    """Construct a Slater determinant

    :parameter boolean optimize_orbitals: make `to_opt` true for orbital parameters
    :parameter array-like twist: The twist to extract from the mean-field object
    :parameter boolean optimize_zeros: optimize coefficients that are zero in the mean-field object
    :returns: slater, to_opt
    """
    
    wf = slater.BosonWF(mol, mf, mc=mc)
    # TODO: update here later
    to_opt = {}
    to_opt["det_coeff"] = np.zeros_like(wf.parameters["det_coeff"], dtype=bool)
    if optimize_determinants:
        to_opt["det_coeff"] = np.ones_like(wf.parameters["det_coeff"], dtype=bool)
        to_opt["det_coeff"][np.argmax(wf.parameters["det_coeff"])] = False
    if optimize_orbitals:
        for k in ["mo_coeff_alpha", "mo_coeff_beta"]:
            to_opt[k] = np.ones(wf.parameters[k].shape, dtype=bool)
            if not optimize_zeros:
                to_opt[k][np.abs(gpu.asnumpy(wf.parameters[k])) < epsilon] = False
    
    return wf, to_opt


# def generate_boson_jastrow(mol, ion_cusp=None, na=4, nb=3, rcut=None):
#     """
#     Default 2-body jastrow from QWalk,

#     :parameter boolean ion_cusp: add an extra term to satisfy electron-ion cusp.
#     :returns: jastrow, to_opt
#     """
#     if ion_cusp == False:
#         ion_cusp = []
#         if not mol.has_ecp():
#             print("Warning: using neither ECP nor ion_cusp")
#     elif ion_cusp == True:
#         ion_cusp = list(mol._basis.keys())
#         if mol.has_ecp():
#             print("Warning: using both ECP and ion_cusp")
#     elif ion_cusp is None:
#         ion_cusp = [l for l in mol._basis.keys() if l not in mol._ecp.keys()]
#     else:
#         assert isinstance(ion_cusp, list)
#     abasis, bbasis = default_jastrow_basis(mol, len(ion_cusp) > 0, na, nb, rcut)
#     jastrow = jastrowspin.JastrowSpin(mol, a_basis=abasis, b_basis=bbasis)
#     if len(ion_cusp) > 0:
#         coefs = mol.atom_charges().copy()
#         coefs[[l[0] not in ion_cusp for l in mol._atom]] = 0.0
#         jastrow.parameters["acoeff"][:, 0, :] = gpu.cp.asarray(coefs[:, None])
#     jastrow.parameters["bcoeff"][0, [0, 1, 2]] = gpu.cp.array([-0.25, -0.50, -0.25])

#     to_opt = {"acoeff": np.ones(jastrow.parameters["acoeff"].shape).astype(bool)}
#     if len(ion_cusp) > 0:
#         to_opt["acoeff"][:, 0, :] = False  # Cusp conditions
#     to_opt["bcoeff"] = np.ones(jastrow.parameters["bcoeff"].shape).astype(bool)
#     to_opt["bcoeff"][0, [0, 1, 2]] = False  # Cusp conditions
#     return jastrow, to_opt


def generate_boson_wf(
    mol, mf, jastrow=generate_jastrow, jastrow_kws=None, slater_kws=None, mc = None, 
):
    """
    """
    if jastrow_kws is None:
        jastrow_kws = {}

    if slater_kws is None:
        slater_kws = {}
    if jastrow == None:
        wf, to_opt1 = generate_boson(mol, mf, mc=mc, **slater_kws)
        to_opt = {"wf1" + k: v for k, v in to_opt1.items()}
    else:        
        no_ei = False
        if 'ei' in jastrow_kws.keys():
            if jastrow_kws['ei'] == False:
                no_ei = True
            del jastrow_kws['ei']

        if not isinstance(jastrow, list):
            jastrow = [jastrow]
            jastrow_kws = [jastrow_kws]

        wf1, to_opt1 = generate_boson(mol, mf, mc=mc, **slater_kws)
        to_opt = {"wf1" + k: v for k, v in to_opt1.items()}

        pack = [jast(mol, **kw) for jast, kw in zip(jastrow, jastrow_kws)]
        wfs = [p[0] for p in pack]
        to_opts = [p[1] for p in pack]
        
        # Do not optimize e-i jastrows
        if no_ei:
            to_opts[0]['bcoeff']*=False

        wf = multiplywf.MultiplyWF(wf1, *wfs)
        for i, to_opt2 in enumerate(to_opts):
            to_opt.update({f"wf{i+2}" + k: v for k, v in to_opt2.items()})
    return wf, to_opt

