import os
import sys
import subprocess
import rcs


# find the test dir in reaction coordinates to create files there
# ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__], capture_output=True).stdout
ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n')
# content of mdp file
MDPSTRING = """
; Running Parameters
; ===================================================================
integrator    = md
nsteps        = 50
dt            = 0.002

; Output control
; ===================================================================
nstxout-compressed = 200

; Constraints
; ===================================================================
constraints     = hbonds            ; standard spc/e: 3 lengths -> also angle fixed

; neighbor searching
; ===================================================================
ns_type         = grid

cutoff-scheme   = verlet            ; standard cutoff scheme
rlist           = 0.9               ; cut-off distance for the short-range neighbor list - with Verlet ignored!
pbc             = xyz               ; Periodic Boundary Conditions (yes/no)
periodic-molecules = yes

; Electrostatics
; ===================================================================
coulombtype     = pme
rcoulomb        = 0.9               ; distance for the Coulomb cut-off
fourierspacing  = 0.1               ; This should give an accuracy of about 2-3*10^-4
pme-order       = 4;

; VdW
; ===================================================================
vdw-type        = Cut-off
vdw-modifier    = Potential-shift-Verlet
DispCorr        = no
rvdw            = 0.9               ; distance for the Van-der-Whaal cut-off
"""


def plumed_dump_masscharge(topology, mcfile, overwrite=False):
    """ To compute a reaction coordinate with plumed --driver, plumed might need the
    mass and charges of all atoms. This is available to plumed while directly interfacing
    with the gromacs during a simulation and can be dumped with plumed. For this, we
    set up a very short simulation from a topology file. Accordingly, the procedure
    only works when gromacs can convert the topology with pdb2gmx.
    Arguments:
        topology (str): path to a .pdb topology file.
        mcfile (str): path were the masscharge file is saved.
        overwrite (bool): overwrite mcfile; default: False.
    """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert os.path.isfile(topology)
    if not overwrite and os.path.exists(mcfile):
        raise FileExistsError('mcfile exists.')
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    # compute hb4 and other stuff with gmx helix
    subprocess.run(
        ['gmx', 'pdb2gmx', '-f', topology, '-ff', 'amber03', '-water', 'spc'],
        check=True,
        capture_output=True
    )
    with open('md.mdp', 'w') as fh:
        fh.write(MDPSTRING)
    subprocess.run([
        'gmx', 'grompp', '-f', 'md.mdp', '-o', 'md.tpr'],
        check=True,
        capture_output=True
    )
    with open('plumed.dat', 'w') as fh:
        fh.write(f'DUMPMASSCHARGE FILE={mcfile}')
    subprocess.run([
        'gmx', 'mdrun', '-deffnm', 'md', '--plumed', 'plumed.dat'],
        check=True,
        capture_output=True
    )
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)


if __name__ == '__main__':
    plumed_dump_masscharge(sys.argv[1], sys.argv[2])
