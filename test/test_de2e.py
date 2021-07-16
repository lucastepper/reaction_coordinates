import os
import subprocess
import pytest
import mdtraj as md
import numpy as np
import rcs


# find the test dir in reaction coordinates to create files there
ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n') + 'test/'


def compute_de2e_gmx(xtc_file, tpr_file, ndx_start, ndx_end):
    """ Compute the de2e reaction coordinate using gromacs. Creates a temp folder
    in ROOTDIR to save the gromacs gyrate results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert np.all([os.path.isfile(ROOTDIR + x) for x in [xtc_file, tpr_file, ndx_start, ndx_end]])
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    output_names = ['com_start.xvg', 'com_end.xvg']
    for input_name, output_name in zip([ndx_start, ndx_end], output_names):
        subprocess.run([
            'gmx', 'traj',
            '-f', ROOTDIR + xtc_file,
            '-s', ROOTDIR + tpr_file,
            '-n', ROOTDIR + input_name,
            '-ox', output_name,
            '-com'],
            check=True,
            capture_output=True,
        )
    # load to numpy array
    com_start = rcs.load_xvg(output_names[0]).astype(np.float32)[:, 1 : 4]
    com_end = rcs.load_xvg(output_names[1]).astype(np.float32)[:, 1 : 4]
    de2e = np.linalg.norm(com_start - com_end, axis=-1)
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return de2e


def compute_de2e_plumed(de2e_obj, xtc_file, mc_file):
    """ Compute the de2e reaction coordinate using plumed driver. Creates a temp folder
    in ROOTDIR to save the plumed results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert np.all([os.path.isfile(ROOTDIR + x) for x in [xtc_file, mc_file]])
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    de2e_obj.write_plumed('plumed.dat', rc_file='de2e_plumed.xvg')
    subprocess.run([
         'plumed', 'driver',
         '--mf_xtc', ROOTDIR + xtc_file,
         '--plumed', 'plumed.dat',
         '--mc', ROOTDIR + mc_file],
        check=True,
    )
    de2e = rcs.load_xvg('de2e_plumed.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return de2e.astype(np.float32)[:, 1]


@pytest.mark.parametrize(
    'xtc_file, tpr_file, ndx_start, ndx_end',
    [
        ('ala9/ala9.xtc', 'ala9/ala9.tpr', 'ala9/ala9_residue1.ndx', 'ala9/ala9_residue9.ndx')
    ],
)
def test_de2e_against_gmx(xtc_file, tpr_file, ndx_start, ndx_end):
    """ Test the computation of the de2e reaction coordinate against
    the result of gromacs. """
    traj = md.load(ROOTDIR + xtc_file, top=ROOTDIR + xtc_file.strip('.xtc') + '.pdb')
    traj.make_molecules_whole()
    de2e = rcs.DE2E(traj)
    de2e_gmx = compute_de2e_gmx(xtc_file, tpr_file, ndx_start, ndx_end)
    np.testing.assert_allclose(de2e.compute(), de2e_gmx, atol=5e-5)


@pytest.mark.parametrize(
    'xtc_file',
    [
        ('ala9/ala9.xtc')
    ],
)
def test_de2e_against_plumed(xtc_file):
    """ Test the computation of the de2e reaction coordinate against
    the result of plumed driver. """
    traj = md.load(ROOTDIR + xtc_file, top=ROOTDIR + xtc_file.strip('.xtc') + '.pdb')
    traj.make_molecules_whole()
    de2e = rcs.DE2E(traj)
    compute_de2e_plumed(de2e, xtc_file, xtc_file.strip('.xtc') + '_mcfile.txt')
