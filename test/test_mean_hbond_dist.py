import os
import subprocess
import mdtraj as md
import numpy as np
import rcs


# find the test dir in reaction coordinates to create files there
# ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__], capture_output=True).stdout
ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n') + 'test/'


def rc_hbonds(chain1, chain2, type_):
    """ Reference implementation for NFGAILS rc. """
    if type_ == 'inner':
        residue_interactions = [('LEU', 'GLY'), ('ALA', 'ILE'), ('PHE', 'SER')]
    elif type_ == 'outer':
        residue_interactions = [('PHE', 'ILE'), ('ALA', 'GLY'), ('LEU', 'ASN')]
    else:
        raise ValueError('type_ not understood.')
    rc = np.zeros(len(chain1.xyz))
    top1 = chain1.topology
    top2 = chain2.topology
    for res1, res2 in residue_interactions:
        res1_o = chain1.xyz[:, top1.select(f'resname {res1} and backbone and name O')]
        res1_n = chain1.xyz[:, top1.select(f'resname {res1} and backbone and name N')]
        res2_o = chain2.xyz[:, top2.select(f'resname {res2} and backbone and name O')]
        res2_n = chain2.xyz[:, top2.select(f'resname {res2} and backbone and name N')]
        hb1 = np.linalg.norm(res1_n - res2_o, axis=-1)
        hb2 = np.linalg.norm(res1_o - res2_n, axis=-1)
        rc += hb1.flatten() + hb2.flatten()
    rc /= 2 * len(residue_interactions)
    return rc


def compute_meanhbonddist_plumed(rc_obj, xtcfile):
    """ Compute the reaction coordinate using plumed driver. Creates a temp folder
    in ROOTDIR to save the plumed results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert os.path.isfile(xtcfile)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    rc_obj.write_plumed('plumed.dat', rc_file='rc.xvg')
    subprocess.run([
         'plumed', 'driver',
         '--mf_xtc', ROOTDIR + xtcfile,
         '--plumed', 'plumed.dat'],
        check=True,
        capture_output=True,
    )
    rc = rcs.load_xvg('rc.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return rc.astype(np.float32)[:, 1]


def test_against_ref():
    """ Check that ref gives the same result. """
    # load data
    traj = md.load('nfgails/nfgails_long.pdb')
    top = traj.topology
    chain1 = md.load('nfgails/nfgails_long.pdb', atom_indices=top.select('chainid 0'))
    chain2 = md.load('nfgails/nfgails_long.pdb', atom_indices=top.select('chainid 1'))
    chain3 = md.load('nfgails/nfgails_long.pdb', atom_indices=top.select('chainid 2'))
    # make rc classes
    rc1 = rcs.MeanHbondDistance(traj, resnames=[('LEU', 'GLY'), ('ALA', 'ILE'), ('PHE', 'SER')], chains=(0, 1))
    rc2 = rcs.MeanHbondDistance(traj, resnames=[('PHE', 'ILE'), ('ALA', 'GLY'), ('LEU', 'ASN')], chains=(0, 1))
    rc3 = rcs.MeanHbondDistance(traj, resnames=[('LEU', 'GLY'), ('ALA', 'ILE'), ('PHE', 'SER')], chains=(1, 2))
    rc4 = rcs.MeanHbondDistance(traj, resnames=[('PHE', 'ILE'), ('ALA', 'GLY'), ('LEU', 'ASN')], chains=(1, 2))
    # test
    assert rc_hbonds(chain1, chain2, type_='inner') - rc1.compute() < 1e-7
    assert rc_hbonds(chain1, chain2, type_='outer') - rc2.compute() < 1e-7
    assert rc_hbonds(chain2, chain3, type_='inner') - rc3.compute() < 1e-7
    assert rc_hbonds(chain2, chain3, type_='outer') - rc4.compute() < 1e-7


def test_against_plumed():
    """ Check that plumed gives the same result. """
    traj = md.load('nfgails/nfgails.xtc', top='nfgails/nfgails.pdb')
    rc1 = rcs.MeanHbondDistance(traj, resnames=[('LEU', 'GLY'), ('ALA', 'ILE'), ('PHE', 'SER')], chains=(0, 1))
    rc2 = rcs.MeanHbondDistance(traj, resnames=[('PHE', 'ILE'), ('ALA', 'GLY'), ('LEU', 'ASN')], chains=(0, 1))
    rc3 = rcs.MeanHbondDistance(traj, resnames=[('LEU', 'GLY'), ('ALA', 'ILE'), ('PHE', 'SER')], chains=(1, 2))
    rc4 = rcs.MeanHbondDistance(traj, resnames=[('PHE', 'ILE'), ('ALA', 'GLY'), ('LEU', 'ASN')], chains=(1, 2))
    # test
    np.testing.assert_allclose(compute_meanhbonddist_plumed(rc1, 'nfgails/nfgails.xtc'), rc1.compute(), atol=1e-6)
    np.testing.assert_allclose(compute_meanhbonddist_plumed(rc2, 'nfgails/nfgails.xtc'), rc2.compute(), atol=1e-6)
    np.testing.assert_allclose(compute_meanhbonddist_plumed(rc3, 'nfgails/nfgails.xtc'), rc3.compute(), atol=1e-6)
    np.testing.assert_allclose(compute_meanhbonddist_plumed(rc4, 'nfgails/nfgails.xtc'), rc4.compute(), atol=1e-6)
