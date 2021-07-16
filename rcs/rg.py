from copy import deepcopy
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
        super().__init__()
        self.name = 'rg'
        self.traj = traj
        self.selection = selection

    def _get_com_pos_masses_idxs(self):
        """ Get center of mass, underlying positions and masses for compute and plot. """
        selection_idxs = self.traj.topology.select(self.selection)
        masses = np.array([x.element.mass for x in self.traj.topology.atoms if x.index in selection_idxs])
        pos = self.traj.xyz[:, selection_idxs, :]
        com = (pos * np.expand_dims(masses, axis=(0, 2))).sum(axis=1) / masses.sum()
        return com, pos, masses, selection_idxs

    def compute(self):
        """ Computes the mass-weighted radius of gyration for every frame. """
        com, pos, masses, __ = self._get_com_pos_masses_idxs()
        com = com.reshape(-1, 1, 3)
        rg = np.sqrt((np.linalg.norm(pos - com, axis=-1) ** 2 * masses).sum(axis=-1) / masses.sum())
        return rg

    def plot(self, **kwargs):
        """ Plot the radius of gyration. Use an quick hack where you write the
        com position into the positions of the first hydrogen in the molecule,
        assuming that hydrogens wont be rendered.
        For kwargs documentation, see the documentation for self.get_view
        and self.add_distances.
        """
        # set white to default
        if 'label_color' not in kwargs:
            kwargs['label_color'] = 'white'
        com, __, __, idxs = self._get_com_pos_masses_idxs()
        h_to_replace = self.traj.topology.select('mass < 2')[0]
        traj_copy = deepcopy(self.traj)
        traj_copy.xyz[:, h_to_replace, :] = com
        view = self.get_view(traj_copy, **{k: v for k, v in kwargs.items() if k in self.kwargs_get_view})
        self.add_distances(
            view,
            [[h_to_replace, idx] for idx in idxs],
            **{k: v for k, v in kwargs.items() if k in self.kwargs_add_distances},
        )
        return view

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
