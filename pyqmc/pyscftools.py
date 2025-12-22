import pyscf
import pyscf.pbc
import pyscf.mcscf
import h5py
import json
import numpy as np

def recover_pyscf(chkfile, ci_checkfile=None, cancel_outputs=True):
    """Generate pyscf objects from a pyscf checkfile, in a way that is easy to use for pyqmc. The chkfile should be saved by setting mf.chkfile in a pyscf SCF object.

    It is recommended to write and recover the objects, rather than trying to use pyscf objects directly when dask parallelization is being used, since by default the pyscf objects contain unserializable objects. (this may be changed in the future)

    `cancel_outputs` will set the outputs of the objects to None. You may need to set `cancel_outputs=False` if you are using this to input to other pyscf functions.

    Typical usage::

        mol, mf = recover_pyscf("dft.hdf5")

    :param chkfile: The filename to read from.
    :type chkfile: string
    :return: mol, mf
    :rtype: pyscf Mole, SCF objects"""

    with h5py.File(chkfile, "r") as f:
        periodic = "a" in json.loads(f["mol"][()]).keys()

    if not periodic:
        mol = pyscf.lib.chkfile.load_mol(chkfile)
        with h5py.File(chkfile, "r") as f:
            if "mo_occ" in f["/scf"].keys():
                mo_occ_shape = f["scf/mo_occ"].shape
            elif "mo_occ__from_list__" in f["/scf"].keys():
                unrestricted = False
                if len(f["/scf/mo_occ__from_list__/"].keys()) == 2:
                    unrestricted = True
                    mo_occ_shape = [f["/scf/mo_occ__from_list__/000000"].shape[0], f["/scf/mo_occ__from_list__/000001"].shape[0]]
                else:
                    mo_occ_shape = [f["/scf/mo_occ__from_list__/000000"].shape[0]]
            else:
                raise Exception("Couldn't determine type from chkfile")
        
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if len(mo_occ_shape) == 2:
            mf = pyscf.scf.UHF(mol)
        elif len(mo_occ_shape) == 1:
            mf = pyscf.scf.ROHF(mol) if mol.spin != 0 else pyscf.scf.RHF(mol)
        else:
            raise Exception("Couldn't determine type from chkfile")
    else:
        mol = pyscf.pbc.lib.chkfile.load_cell(chkfile)
        with h5py.File(chkfile, "r") as f:
            has_kpts = "mo_occ__from_list__" in f["/scf"].keys()
            if has_kpts:
                rhf = "000000" in f["/scf/mo_occ__from_list__/"].keys()
            else:
                rhf = len(f["/scf/mo_occ"].shape) == 1
        if cancel_outputs:
            mol.output = None
            mol.stdout = None
        if not rhf and has_kpts:
            mf = pyscf.pbc.scf.KUHF(mol)
        elif has_kpts:
            mf = pyscf.pbc.scf.KROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.KRHF(mol)
        elif rhf:
            mf = pyscf.pbc.scf.ROHF(mol) if mol.spin != 0 else pyscf.pbc.scf.RHF(mol)
        else:
            mf = pyscf.pbc.scf.UHF(mol)
    mf.__dict__.update(pyscf.scf.chkfile.load(chkfile, "scf"))
    mf.mo_occ = np.array(mf.mo_occ)
    if ci_checkfile is not None:
        casdict = pyscf.lib.chkfile.load(ci_checkfile, "ci")
        if casdict is None:
            casdict = pyscf.lib.chkfile.load(ci_checkfile, "mcscf")
        with h5py.File(ci_checkfile, "r") as f:
            hci = "ci/_strs" in f.keys()
        if hci:
            mc = pyscf.hci.SCI(mol)
        else:
            # if len(casdict["mo_coeff"].shape) == 3:
            #     mc = pyscf.mcscf.UCASCI(mol, casdict["ncas"], casdict["nelecas"])
            # else:
            mc = pyscf.mcscf.CASCI(mol, casdict["ncas"], casdict["nelecas"])
        
        mc.__dict__.update(casdict)

        return mol, mf, mc
    return mol, mf

def load_mf_inputs_from_hdf5(chkfile, mol=None):
    """
    Load mf_inputs dictionary from HDF5 checkfile if available.
    
    This function checks if the checkfile contains a pre-computed 'mf_inputs' group.
    If found, it loads all the stored data. The 'grids' object is reconstructed
    from the molecule if needed, since it cannot be directly serialized.
    
    Parameters
    ----------
    chkfile : str
        Path to the HDF5 checkfile
    mol : pyscf Mole object, optional
        Molecule object needed to reconstruct grids if not stored properly.
        If None, will try to load from checkfile.
        
    Returns
    -------
    mf_inputs : dict or None
        Dictionary containing mf_inputs if found in checkfile, None otherwise.
        Keys include: 'xc', 'deriv', 'nelec', 'mo_energy', 'mo_occ', 'mo_coeff', 
        'dm', 'rho', 'grids'
    """
    import h5py
    from pyscf import dft
    
    try:
        with h5py.File(chkfile, "r") as f:
            if "mf_inputs" not in f.keys():
                return None
            
            mf_inputs_grp = f["mf_inputs"]
            mf_inputs = {}
            
            # Load datasets (numpy arrays)
            for key in mf_inputs_grp.keys():
                if isinstance(mf_inputs_grp[key], h5py.Dataset):
                    mf_inputs[key] = mf_inputs_grp[key][:]
            
            # Load attributes (scalars)
            for key in mf_inputs_grp.attrs.keys():
                value = mf_inputs_grp.attrs[key]
                # Handle None stored as string
                if value == "None":
                    mf_inputs[key] = None
                elif isinstance(value, (str, bytes)):
                    # Try to convert string representations back to Python objects
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    # Check if it's a tuple representation like "(2, 1)" or "(2,1)"
                    if value.startswith('(') and value.endswith(')'):
                        try:
                            import ast
                            mf_inputs[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If parsing fails, keep as string
                            mf_inputs[key] = value
                    else:
                        mf_inputs[key] = value
                else:
                    mf_inputs[key] = value
            
            # Reconstruct grids object if needed
            # The grids object cannot be directly serialized, so it's stored as a string/repr
            # or we need to reconstruct it from the molecule
            grids_reconstructed = False
            
            # Check if grids_coords and grids_weights are stored separately (preferred method)
            if 'grids_coords' in mf_inputs and 'grids_weights' in mf_inputs:
                if mol is None:
                    mol = pyscf.lib.chkfile.load_mol(chkfile)
                grids = dft.gen_grid.Grids(mol)
                grids.coords = mf_inputs['grids_coords']
                grids.weights = mf_inputs['grids_weights']
                mf_inputs['grids'] = grids
                # Remove temporary keys
                del mf_inputs['grids_coords']
                del mf_inputs['grids_weights']
                grids_reconstructed = True
            elif 'grids' in mf_inputs:
                # If grids was stored as a string/repr (from run_atom.py), we need to reconstruct it
                if isinstance(mf_inputs['grids'], (str, bytes)):
                    # Need mol to reconstruct grids
                    if mol is None:
                        mol = pyscf.lib.chkfile.load_mol(chkfile)
                    
                    # Reconstruct grids from molecule
                    grids = dft.gen_grid.Grids(mol)
                    grids.level = 5  # Default level, matching calculate_mf_density
                    grids.build()
                    mf_inputs['grids'] = grids
                    grids_reconstructed = True
            
            # If grids is still missing, reconstruct it from mol and dm
            if not grids_reconstructed and 'grids' not in mf_inputs:
                if mol is None:
                    mol = pyscf.lib.chkfile.load_mol(chkfile)
                if 'dm' in mf_inputs:
                    # Reconstruct grids using the same method as calculate_mf_density
                    grids = dft.gen_grid.Grids(mol)
                    grids.level = 5  # Default level, matching calculate_mf_density
                    grids.build()
                    mf_inputs['grids'] = grids
                    print("Warning: Reconstructed grids object from molecule (not found in checkfile)")
            
            # Ensure mol is in mf_inputs (needed for compatibility)
            if 'mol' not in mf_inputs:
                if mol is None:
                    mol = pyscf.lib.chkfile.load_mol(chkfile)
                mf_inputs['mol'] = mol
            
            # Convert string xc to proper format if needed
            if 'xc' in mf_inputs and isinstance(mf_inputs['xc'], bytes):
                mf_inputs['xc'] = mf_inputs['xc'].decode('utf-8')
            elif 'xc' in mf_inputs and isinstance(mf_inputs['xc'], str):
                # Already a string, ensure it's uppercase for consistency
                mf_inputs['xc'] = mf_inputs['xc'].upper()
            return mf_inputs
            
    except Exception as e:
        # If anything goes wrong, return None to fall back to old method
        print(f"Warning: Could not load mf_inputs from {chkfile}: {e}")
        return None
