# Wavefunction generation
from pyscf import mcscf, fci, lib 
from pyscf import gto, scf, tools, dft
import api as pyq 
import importlib
import os
import h5py
import pandas as pd
import pdb
import pyscf
import numpy as np
import pyqmc
import scipy
import matplotlib.pyplot as plt
from pyscf.scf.hf import dump_scf_summary
from concurrent.futures import ProcessPoolExecutor
from pyscf.scf.chkfile import dump_scf
print(pyq.__file__)
print(pyscf.__file__)

def run_lda_he(scf_checkfile="he.hdf5"):
    
    print("He atom neutral LDA spin=0")
    # mol = gto.M(atom="He 0. 0. 0.", basis="ccECP_cc-pVDZ", ecp="ccecp", unit='bohr')
    mol = gto.M(atom="He 0. 0. 0.", basis="aug-ccpvqz", unit='bohr')    
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    # dm = mf.init_guess_by_atom()
    # mf.kernel(dm, xc='LDA')
    mf.xc = 'LDA'
    mf.chkfile = scf_checkfile
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_li(scf_checkfile="li.hdf5"):
    print("Li atom neutral LDA spin=1 aug-ccpvqz")
    mol = gto.M(atom="Li 0. 0. 0.", spin = 1, basis="aug-ccpvqz", unit='bohr')    
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()

    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_be(scf_checkfile="be.hdf5"):
    print("Be atom neutral LDA spin=0 aug-ccpvqz")
    mol = gto.M(atom="Be 0. 0. 0.", spin=0, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()

    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_b(scf_checkfile="b.hdf5"):
    print("B atom neutral LDA spin=1 aug-ccpvqz")
    mol = gto.M(atom="B 0. 0. 0.", spin=1, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_c(scf_checkfile="c.hdf5"):
    print("C atom neutral LDA spin=2 aug-ccpvqz")
    mol = gto.M(atom="C 0. 0. 0.", spin=2, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf


def run_lda_n(scf_checkfile="n.hdf5"):
    print("N atom neutral LDA spin=3 aug-ccpvqz")
    mol = gto.M(atom="N 0. 0. 0.", spin=3, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_o(scf_checkfile="o.hdf5"):
    print("O atom neutral LDA spin=2 aug-ccpvqz")
    mol = gto.M(atom="O 0. 0. 0.", spin=2, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf

def run_lda_f(scf_checkfile="f.hdf5"):
    print("F atom neutral LDA spin=1 aug-ccpvqz")
    mol = gto.M(atom="F 0. 0. 0.", spin=1, basis='aug-ccpvqz', unit='bohr')
    # print("HF")
    # mf = scf.UHF(mol)
    # mf.kernel()
    print("LDA")    
    mf = dft.UKS(mol)
    mf.chkfile = scf_checkfile
    mf.xc = 'LDA'
    mf.kernel()
    opt_checkfile = scf_checkfile.split('.hdf5')[0]+'-sj.hdf5'
    return scf_checkfile, opt_checkfile, mf
    
def run_casscf(scf_checkfile, ci_checkfile):
    cell, mf = pyq.recover_pyscf(scf_checkfile, cancel_outputs=False)
    mc = mcscf.CASSCF(mf,2,2)
    mc.chkfile = ci_checkfile
    mc.kernel()
    with h5py.File(mc.chkfile, "a") as f:
        print("Available output from CASSCF:", f["mcscf"].keys())
        f["mcscf/nelecas"] = list(mc.nelecas)
        f["mcscf/ci"] = mc.ci
    return mc

def run_casci(scf_chkfile, ncas = None, nroots=None, nelecas = None, dryrun= False):
    rootname = scf_chkfile.split('.hdf5')[0]
    ci_chkfile = rootname+'-ci.hdf5'
    cell, mf = pyq.recover_pyscf(scf_chkfile, cancel_outputs=False)
    # ncas: orbitals
    # nelecas: electrons
    if nelecas is None:
        nelecas = mf.nelec
    try:
        nelecas_str='_'.join([str(x) for x in nelecas])
    except:
        nelecas_str=str(nelecas)
    opt_chkfile = rootname+'_opt_cas_'+str(ncas)+'_nelecas_'+str(nelecas_str)+'.hdf5'
    vmc_chkfile = rootname+'_vmc_cas_'+str(ncas)+'_nelecas_'+str(nelecas_str)+'.hdf5'
    if dryrun: 
        return ci_chkfile, None, opt_chkfile, vmc_chkfile            
    else:
        print('CASCI nelecas up/down', nelecas)
        mc = mcscf.CASCI(mf, ncas, nelecas)
        if nroots is not None:
            mc.fcisolver.nroots = nroots
        for fname in [ci_chkfile]:
            if os.path.isfile(fname):
                os.remove(fname)   
        mc.kernel()
        # print(mc.__dict__.keys())
        with h5py.File(ci_chkfile, "a") as f:
            f.create_group("ci")
            f["ci/ncas"] = mc.ncas
            f["ci/ncore"] = mc.ncore
            f["ci/nelecas"] = list(mc.nelecas)
            f["ci/fci"] = mc.ci
            f["ci/ci"] = mc.ci
            f["ci/mo_coeff"] = mc.mo_coeff
            f["ci/mc_mo_energy"] = mc.mo_energy
            f["ci/mf_mo_energy"] = mf.mo_energy
            f["ci/mo_occ"] = mf.mo_occ
            print("Available output from CASCI:", f["ci"].keys())
        return ci_chkfile, mc, opt_chkfile, vmc_chkfile

def make_wf_object(scf_checkfile, ci_checkfile):
    mol, mf, mc = pyq.recover_pyscf(scf_checkfile, ci_checkfile=ci_checkfile)
    wf, _ = pyq.generate_wf(mol, mf, mc=mc)
    return wf

def stat_qmc(etot, filename):
    block_sizes = np.linspace(2,len(etot[discard:])//16, 10, dtype=int)
    reblocks = [len(etot)//s for s in block_sizes]

    plt.figure()
    df = pd.DataFrame([pyq.read_mc_output(filename, warmup=discard, reblock=reblock) for reblock in reblocks])
    df['block_size'] = block_sizes
    plt.plot("block_size",'energytotal_err',data=df, marker='o')
    plt.xlabel("Block size")
    plt.ylabel("Estimated uncertainty (Ha)")

def reblock(e, discard, reblock_size):
    e = e[discard:]
    vals = pyqmc.reblock.reblock(e,int(len(e)/reblock_size))
    
    e_m  = np.mean(vals, axis=0)
    e_d  = scipy.stats.sem(vals, axis=0)    
    return e, e_m, e_d

# scf_checkfile = "scf.hdf5"
# mf = run_scf(scf_checkfile)
# mf_lda = run_lda_he(scf_checkfile)

# ci_checkfile, mc = run_casci(scf_checkfile)
