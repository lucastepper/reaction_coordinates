import numpy as np
from rcs.reaction_coordinate_base import ReactionCoordinate


class HB4(ReactionCoordinate):
    """ Computes the hb4 RC, measuring the formation of alpha helices.
    The RC is defined as the average distance between oxygens on the
    N-terminus and nitrogens on de C-terminus separated by 4 residues.
    Axstart and Axend are the same as with gromacs helix.
    """
    def __init__(self, traj, axstart, axend):
        """ Constructor Arguments:
            traj (mdtraj.Trajectory): trajectory
            axstart (int): first residue to consider, starting at 1
            axend (int): last residue to consider, starting at 1, inclusive
        """
        self.name = 'hb4'
        self.traj = traj
        self.axstart = axstart
        self.axend = axend

    def get_indexes(self):
        """ Find all indexes for distances involved in hb4 RC. """
        hb4_idxs = []
        top = self.traj.topology
        for i in range(self.axstart - 1, self.axend - 4):
            idx_n = top.select(f'resid {i} and backbone and type O')
            idx_o = top.select(f'resid {i + 4} and backbone and type N')
            hb4_idxs.append((idx_n, idx_o))
        return hb4_idxs

    def compute(self):
        """ Computes the hb4 RC given traj, axstart, axend from constructor. """
        hb4 = np.zeros(len(self.traj))
        hb4_idxs = self.get_indexes()
        for idx1, idx2 in hb4_idxs:
            hb4 += np.linalg.norm(self.traj.xyz[:, idx1, :] - self.traj.xyz[:, idx2, :], axis=-1).flatten()
        return hb4 / len(hb4_idxs)

    def plot(self, view=None, frames=(0, 1), **kwargs):
        """ Plot the HB4 reaction coordinate for the trajectory stored in this class.
        The protein is plotted as a cartoon together with the backbone atoms.
        The N-O distances that comprise the HB4 reaction coordinate are plotted
        as dashed lines.
        Arguments:
            view (nglview.NGLWidget): scence to render the image into;
                if None instantiates a new one; default None
            frames (tuple len=2): frames to plot, set is inclusive left, exclusive right
                default (0, 1); which is the first frame
            kwargs: additional settings for plotting, these are:
            opacity (float): opacity for rendering the backbone atoms
            color (list, len=3): RBG code for color of HB4 bonds rendered; default [1, 0, 1]
            thickness (float): thickness of the HB4 bonds rendered; default 0.1
            n_dash (int): number of dashes of the HB4 bond; default 11
            factor (float): conversion factor for the traj positions. nglview uses Angstrom
                gromacs nm; default 10 (nm -> A)
        """
        if not view:
            view = self.get_view(frames=frames, **{k: v for k, v in kwargs.items() if k in ['opacity']})
        hb4_idxs = self.get_indexes()
        for idx1, idx2 in hb4_idxs:
            self.add_distance(
                view,
                self.traj.xyz[frames[0], idx1, :],
                self.traj.xyz[frames[0], idx2, :],
                **{k: v for k, v in kwargs.items() if k in ['color', 'thickness', 'n_dash', 'factor']}
            )
        return view


    def get_lines_plumed(self):
        """ Get the lines needed for the input file to compute
            the hb4 RC with plumed programm.
        """
        hb4_idxs = self.get_indexes()
        lines = []
        for i, (idx1, idx2) in enumerate(hb4_idxs):
            lines.append(f'hb4d{i + 1}: DISTANCE ATOMS={int(idx1) + 1},{int(idx2) + 1} NOPBC \n')
        lines.append(' \n')
        lines.append(f'{self.name}: CUSTOM ...  \n')
        lines.append('ARG={}  \n'.format(','.join([f'hb4d{i + 1}' for i in range(len(hb4_idxs))])))
        lines.append('VAR={}  \n'.format(','.join([f'a{i}' for i in range(len(hb4_idxs))])))
        lines.append(
            'FUNC={'
            + '({}) / '.format(' + '.join([f'a{i}' for i in range(len(hb4_idxs))]))
            + '%i} \n'%len(hb4_idxs)
        )
        lines.append('PERIODIC=NO \n')
        lines.append('... \n')
        lines.append(' \n')
        return lines