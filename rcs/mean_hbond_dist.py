import numpy as np
import mdtraj as md
from rcs.reaction_coordinate_base import ReactionCoordinate


class MeanHbondDistance(ReactionCoordinate):
    """ Computes a mean distance between the two hydrogen bonds a contact
    between two residues can form, measured by the distance between
    N1 -> O2 and N2 -> O1 of the backbone.
    """
    def __init__(self, traj, resnames=(), chains=(), name='mean_hbond_dist', top=None):
        """ Constructor Arguments:
            traj (mdtraj.Trajectory): trajectory
            resnames (list of lists): list of lists, containing pairs of contacts
            chains (list of list or str): list containing chain1 and chain 2 for 
                contacts, if all the same, or list of lists.
            name (str): name for the reaction coordinate; default: mean_hbond_dist
        """
        super().__init__(top=top)
        self.name = name
        self.traj = traj
        self._contacts = []
        if isinstance(chains[0], int) and isinstance(chains[1], int):
            chains = [(chains[0], chains[1]) for __ in range(len(resnames))]
        for chain, resname in zip(chains, resnames):
            self.add_contact(resname1=resname[0], resname2=resname[1], chainid1=chain[0], chainid2=chain[1])

    def add_contact(self, resid1=None, resid2=None, resname1=None, resname2=None, chainid1=None, chainid2=None):
        """ Indexes 0-based. """
        # check that residues 1 and 2 have an identifier
        idxs1 = self._identify_residue(resid1, resname1, chainid1, 'first')
        idxs2 = self._identify_residue(resid2, resname2, chainid2, 'second')
        # oxygen then nitrogen index returned, get N->O and O->N indexes
        self._contacts.extend([
            (idxs1[0], idxs2[1]),
            (idxs1[1], idxs2[0]),
        ])

    def _identify_residue(self, resid, resname, chainid, name):
        """ Check that given the description for the residue, a single
            Oxygen and Nitrogen can be identified and return the indices. """
        if not resid and not resname:
            raise ValueError(f'Please identify the {name} residue involved in contact either by name or by id.')
        select_str = ''
        if resname is not None:
            select_str += f'resname {resname} and '
        if resid is not None:
            select_str += f'residue {resid} and '
        if chainid is not None:
            select_str += f'chainid {chainid} and '
        select_str += 'backbone and '
        idxs_O = self.traj.topology.select(select_str + 'name O')
        idxs_N = self.traj.topology.select(select_str + 'name N')
        if len(idxs_O) > 1 or len(idxs_O) == 0:
            print(f'The {name} residues identifier for {resname} in chain {chainid} returned {len(idxs_O)} residues.')
            print(select_str)
        idxs = [int(idxs_O), int(idxs_N)]
        return idxs

    def plot(self, view=None, **kwargs):
        """
        TODOODODO
        """
        if not view:
            view = self.get_view(self.traj, **{k: v for k, v in kwargs.items() if k in ['opacity']})
        for __ in self._contacts:
            self.add_distances(
                view,
                self._contacts,
                **{k: v for k, v in kwargs.items() if k in ['color', 'label_color']},
            )
        return view

    def compute(self):
        return md.compute_distances(self.traj, np.array(self._contacts)).mean(-1)

    def get_lines_plumed(self):
        """ Get the lines needed for the input file to compute
            the RC with plumed programm.
        """
        lines = []
        for i, (idx1, idx2) in enumerate(self._contacts):
            lines.append(f'cont{i + 1}: DISTANCE ATOMS={int(idx1) + 1},{int(idx2) + 1} NOPBC \n')
        lines.append(' \n')
        lines.append(f'{self.name}: CUSTOM ...  \n')
        lines.append('ARG={}  \n'.format(','.join([f'cont{i + 1}' for i in range(len(self._contacts))])))
        lines.append('VAR={}  \n'.format(','.join([f'a{i}' for i in range(len(self._contacts))])))
        lines.append(
            'FUNC={'
            + '({}) / '.format(' + '.join([f'a{i}' for i in range(len(self._contacts))]))
            + '%i} \n'%len(self._contacts)
        )
        lines.append('PERIODIC=NO \n')
        lines.append('... \n')
        lines.append(' \n')
        return lines
