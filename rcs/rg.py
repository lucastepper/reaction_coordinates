import numpy as np
from rcs.reaction_coordinate_base import ReactionCoordinate


class RG(ReactionCoordinate):
    """ Computes the radius of gyration RC, measuring the compactness
    of the selected molecule. RG is mass-weighted.
    """
    def __init__(self, traj, selection):
        """ Constructor Arguments:
            traj (mdtraj.Trajectory): trajectory
            selection (str): selection for the set of atoms to compute
                rg from. Selection language from mdtraj, see reference:
                https://mdtraj.org/1.9.4/atom_selection.html
        """
        self.name = 'rg'
        self.traj = traj
        self.selection = selection

    def compute(self):
        """ Computes the mass-weighted radius of gyration for every frame. """
        selection_idxs = self.traj.topology.select(self.selection)
        masses = np.array([x.element.mass for x in self.traj.topology.atoms if x.index in selection_idxs])
        pos = self.traj.xyz[:, selection_idxs, :]
        com = (pos * np.expand_dims(masses, axis=(0, 2))).sum(axis=1) / masses.sum()
        com = com.reshape(-1, 1, 3)
        rg = np.sqrt((np.linalg.norm(pos - com, axis=-1) ** 2 * masses).sum(axis=-1) / masses.sum())
        return rg

    def get_lines_plumed(self):
        """ Get the lines needed for the input file to compute
            the rg RC with plumed programm.
        """
        selection_idxs = self.traj.topology.select(self.selection)
        # remember that plumed starts to count at one
        atoms_str = ','.join([str(x + 1) for x in selection_idxs])
        lines = []
        lines.append(f'rg: GYRATION MASS_WEIGHTED ATOMS={atoms_str}')
        lines.append(' \n')
        return lines
