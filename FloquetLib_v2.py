###########################################################################
#
# Mofidied Two-Atom Rydberg Floquet Library
# Deniz Kurdak
# April 2025
#
# Original codebase written by Patrick Banner
#
###########################################################################
#
# This library is a set of functions for modeling Rydberg atoms
# under the simultaneous influence of DC magnetic fields, DC and AC electric
# fields (microwaves or optical fields in the latter case), and dipole-dipole
# interactions. To do it all we're using Floquet formalism in the two-atom basis.

# Import statements
from arc import *
import numpy as np
from scipy import sparse as sp
from scipy.constants import physical_constants
from scipy import optimize as opt
from matplotlib import pyplot as plt
import time
import os
import h5py as h5

# Key global variables
atom = Rubidium87()

hbar = C_h/(2*np.pi)
a0 = physical_constants["Bohr radius"][0]
pi = np.pi
epsilon0 = 8.854e-12

from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines

def getAlphaCMap(r,g,b, aStart=0, aEnd=1, aExp=1.0):
    '''
    This function makes a color map for plotting purposes, with
    the variation being in opacity, from the color given by r,g,b
    at the opacity specified by aEnd to the same color but at the
    opacity specified by aStart. You may find that a different
    function for opacity variation produces clearer plots, so you
    can scale the opacity function as opacity^aExp.
    Inputs: r,g,b (required)
            aStart=0, aEnd=1, aExp=1.0 (optional)
    Outputs: ListedColormap type with 256 colors
    '''
    N = 256
    vals = np.ones((N, 4))
    vals[:,0] = r; vals[:,1] = g; vals[:,2] = b
    vals[:, 3] = np.linspace(aStart, aEnd, N)**aExp
    return ListedColormap(vals)
def getEdgeCMap(r,g,b, edge=0.05):
    '''
    This function makes a color map for plotting purposes. For all
    inputs above the parameter edge, the colormap gives the color
    (r,g,b); for all inputs below the edge, it gives total transparency.
    Inputs: r,g,b (required)
            edge=0.05 (optional)
    Outputs: ListedColormap type with 256 colors
    '''
    N = 256
    vals = np.ones((N, 4))
    vals[:,0] = r; vals[:,1] = g; vals[:,2] = b
    edgeN = int(round(N*edge))
    vals[0:edgeN, 3] = 0
    vals[edgeN:, 3] = 1
    return ListedColormap(vals)

# Utility function for getting atomic levels from parameters
def getRange(a,b,s=1): return np.arange(a,b+1,s)

def generate_shades(base_color, num_colors, target_color=(1.0, 1.0, 1.0)):
    """
    Generates a numpy array of RGB tuples, interpolating from base_color to a target_color.

    Parameters:
        base_color (tuple): Starting RGB color as (R, G, B), values from 0.0 to 1.0.
        num_colors (int): Number of colors to generate.
        target_color (tuple): Target RGB color as (R, G, B), values from 0.0 to 1.0.

    Returns:
        numpy.ndarray: Array of shape (num_colors, 3) with RGB values normalized between 0 and 1.
    """
    base_color = np.array(base_color)
    target_color = np.array(target_color)
    
    # Generate interpolation factors between 0 and 1
    t_values = np.linspace(0, 1, num_colors)
    
    # Interpolate between base_color and target_color
    colors = (1 - t_values[:, None]) * base_color + t_values[:, None] * target_color
    
    return colors


###########################################################################
#
# Atom Classes
#
###########################################################################

class oneAtomState:
    n=0; l=0; j=0; mj=0;
    q=0     # Fourier component for Floquet state

    # Initialization with n, l, j, mj, and optional q_in for the Fourier component
    def __init__(self,n_in,l_in,j_in,mj_in,q_in = 0):
        self.n = int(n_in); self.l = int(l_in); self.j = j_in; self.mj = mj_in
        self.q = int(q_in)
    
    # Convert to printable string
    def __str__(self):
        return "({:.0f}, {:.0f}, {:.1f}, {:.1f}, q={:.0f})".format(self.n, self.l, self.j, self.mj, self.q)
    
    # Convert to Python list (mostly for saving to files)
    def tolist(self):
        return [self.n, self.l, self.j, self.mj, self.q]

    def toKet(self):
        if self.l ==0:
            Ls = 'S';
        elif self.l ==1:
            Ls = 'P';
        elif self.l ==2:
            Ls = 'D';
        return '|{:.0f}{}_{:.1f}, m_J = {:.1f}, q={:.0f}>'.format(self.n, Ls, self.j, self.mj, self.q)

    # Checking equality between two states
    def __eq__(self, other):
        if self.n == other.n and self.l == other.l and self.j == other.j and self.mj == other.mj and self.q == other.q:
            return True
        else:
            return False

class twoAtomState:
    n1=0; ll=0; jl=0; mjl=0;
    n2=0; l2=0; j2=0; mj2=0;
    q=0     # Fourier component for Floquet state

    # Initialization with n, l, j, mj, and optional q_in for the Fourier component
    def __init__(self,n1_in,l1_in,j1_in,mj1_in,n2_in,l2_in,j2_in,mj2_in,q_in = 0):
        self.n1 = int(n1_in); self.l1 = int(l1_in); self.j1 = j1_in; self.mj1 = mj1_in
        self.n2 = int(n2_in); self.l2 = int(l2_in); self.j2 = j2_in; self.mj2 = mj2_in
        self.q = int(q_in)

    # Call as twoAtomState.fromOneAtomStates(s1, s2) where s1, s2 are oneAtomStates
    # Returns the relevant two-atom states
    @classmethod
    def fromOneAtomStates(cls, s1, s2):
        if (s1.q != s2.q):
            raise Exception("The q values of the states must match.")
        return cls(s1.n, s1.l, s1.j, s1.mj, s2.n, s2.l, s2.j, s2.mj, s1.q)
    
    # Convert to printable string
    def __str__(self):
        return "(({:.0f}, {:.0f}, {:.1f}, {:.1f}), ({:.0f}, {:.0f}, {:.1f}, {:.1f}), q={:.0f})".format(self.n1, self.l1, self.j1, self.mj1, self.n2, self.l2, self.j2, self.mj2, self.q)
    
    # Convert to Python list (mostly for saving to files)
    def tolist(self):
        return [self.n1, self.l1, self.j1, self.mj1, self.n2, self.l2, self.j2, self.mj2, self.q]
    
    # Checking equality between two states
    def __eq__(self, other):
        if self.n1 == other.n1 and self.l1 == other.l1 and self.j1 == other.j1 and self.mj1 == other.mj1 and self.n2 == other.n2 and self.l2 == other.l2 and self.j2 == other.j2 and self.mj2 == other.mj2 and self.q == other.q:
            return True
        else:
            return False


###########################################################################
#
# Getting State Lists
#
###########################################################################

def getStateList(nMin, nMax, maxL):
    '''
    This function returns a Python list of one-atom states corresponding
    to the input parameters, i.e. all those states with nMin <= n <= nMax
    and 0 <= l <= maxL.
    '''
    
    stateList = []
    
    for n in getRange(nMin, nMax):
        for l in getRange(0,maxL):
            for j in getRange(abs(l-1/2),l+1/2,1):
                for m_j in getRange(-j,j):
                    stateList.append(oneAtomState(n,l,j,m_j))
                                    
    return stateList

def getOneAtomFloquetStateList(stateList, qMax):
    '''
    Given a list of one-atom states (e.g. from getStateList()), this returns
    a list of oneAtomStates of all the one-atom Floquet states (i.e. with
    correctly-assigned q values). 
    Inputs: stateList, the Python list of oneAtomState objects, and qMax,
            specifying the range of Fourier components to include (the range
            is such that all states with -qMax <= q <= qMax will be included).
    '''
    
    if qMax < 0:
        print("You need to specify qMax >= 0.")
        return None
    
    # Let's make the Floquet state list
    stateNum = len(stateList)
    floquetStateList = []
    for q in getRange(-qMax, qMax):
        for j in range(stateNum):
            floquetStateList.append(oneAtomState(stateList[j].n,stateList[j].l,stateList[j].j,stateList[j].mj,q_in=q))
            
    return floquetStateList

# Given an atomic state list, make the two-atom Floquet states, whose Fourier indices go from -qMax to qMax inclusively
def getTwoAtomFloquetStateList(oneAtomStateList, qMax):
    '''
    Given a list of one-atom states (e.g. from getStateList()), this returns
    a list of twoAtomStates of all the two-atom Floquet states (i.e. with
    correctly-assigned q values, and every choice of two of the one-atom states
    in the input list). 
    Inputs: oneAtomStateList, the Python list of oneAtomState objects, and qMax,
            specifying the range of Fourier components to include (the range
            is such that all states with -qMax <= q <= qMax will be included).
    '''
    
    if qMax < 0:
        print("You need to specify qMax >= 0.")
        return None
    
    # Let's make the Floquet state list
    stateNum = len(oneAtomStateList)
    floquetStateList = []
    for q in getRange(-qMax, qMax):
        for j in range(stateNum):
            for k in range(stateNum):
                floquetStateList.append(twoAtomState(
                    oneAtomStateList[j].n,oneAtomStateList[j].l,oneAtomStateList[j].j,oneAtomStateList[j].mj,
                    oneAtomStateList[k].n,oneAtomStateList[k].l,oneAtomStateList[k].j,oneAtomStateList[k].mj,
                    q_in=q))
            
    return floquetStateList


###########################################################################
#
# Getting Fractions of Single Atom States
#
###########################################################################

def getFracsOfStateSingleAtom(oneAtomStateList, qMax, evecs, soi, stateInds = [], v = [], psfBool=0):
    '''
    This function gets a desired state's overlap with all of the eigenvectors of a system.
    It is primarily used for color mapping in plotting.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects that were the one-atom
                          basis used when generating the input eigenvectors (so this
                          function can reconstruct the basis that those eigenvectors are
                          expressed in)
        qMax: the maximum Fourier component used when generating the input eigenvectors
              (so this function can reconstruct the basis that those eigenvectors are expressed in)
        evecs: the eigenvectors of the system, with which to find the overlap
               NB: this should be a 2D NumPy array whose ROWS are the eigenvectors, e.g.
               evecs[0] is an eigenvector of the system; this is the TRANSPOSE of what
               numpy.linalg.eigh() and scipy.sparse.eigs() return

        The state with which to find the overlap is described jointly by several parameters:
        soi, SAS, stateInds, and v.
        
        soi: the two-atom state info for this function to work with; in conjunction with input SAS
             (see below) (this should be a twoAtomState object)
        SAS: an integer that determines exactly what this function does:
             0: find the overlap with the soi
             ±1: find the overlap with |s1s2> ± |s2s1> where |s1s2> is the soi
             +2: find the overlap with the state (v[0] |s1> + v[1] |s2>) × (v[0] |s1> + v[1] |s2>),
                 where |s1s2> is the soi and v is a keyword argument
             +3: find the overlap with the state given by the keyword argument v, expressed as
                 a vector in the full, two-atom Floquet basis
        stateInds (optional): a list of indices of states to keep in the two-atom Floquet basis; if
                   keeping all of the states, use the empty set [] (default [])
        v (optional unless SAS = +2 or +3):
                If SAS = ± 2, should be a two-item list describing the one-atom state coefficients
                with which to find the overlap. 
                If SAS = +3, should be a (2*qMax+1)*len(oneAtomStateList)**2-length vector of
                state coefficients describing a state in the two-atom Floquet basis with which to
                find the overlap.
        psfBool (optional unless SAS = 2): standing for "product state Floquet Boolean", when SAS = 2
                this Boolean specifies whether the Floquet index of the corresponding two-atom
                states |s1s1>,|s1s2>,|s2s1>,|s2s2> should all be the same as the Floquet index
                of soi, or it should change to account for the s1->s2 transition requiring a photon.
                1 if it should change, 0 by default. If 1, the order in which soi is specified
                matters: the state not requiring a photon to get to should be specified first.
                Note that I'm calling it a Boolean but it's really juust an integer; values like
                2 (for two-photon transitions) or -1 (if you get the order wrong) will work too.
        
    Outputs:
        cs: a len(evecs)-length list containing the overlaps (|<sA|sB>|^2) of every eigenvector with
            the state of intereste described by soi, SAS, stateInds, and v
    '''

    # Get the one-atom Floquet state list
    oneAtomFSL = np.array(getOneAtomFloquetStateList(oneAtomStateList, qMax), dtype=object)
    soiInd = np.where(oneAtomFSL == soi)[0][0]
    v = np.zeros(len(oneAtomFSL)); v[soiInd] = 1

    # Now take the dot product
    cs = np.abs(np.dot(v, np.transpose(evecs)))**2
    # I think this can made into one line because NumPy is smart, but it's fine
    #for i, vec in enumerate(evecs):
    #    cs[i] = np.abs(np.dot(v, vec))**2
    return cs


###########################################################################
#
# DC Electric Field Hamiltonian Construction - Single Atom
#
###########################################################################

def getH_E_DC_singleAtom_base(oneAtomStateList, soi, pol, E = 1):
    '''
    [Primarily a utility method]
    Get the part of the one-atom Hamiltonian containing the DC electric field couplings.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        pol: a 3-element Python list with information about the direction of the 
             electric field vector [x,y,z]; does not need to be normalized
        E (optional): the magnitude of the field (V/cm); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    stateNum = len(oneAtomStateList) # state number
    H = sp.lil_matrix((stateNum, stateNum), dtype='complex_')

    # Normalize the polarization vector, and convert it to the atomic spherical basis
    polN = pol/np.linalg.norm(pol)
    E_pol = [E*complex(polN[0], -polN[1])/np.sqrt(2), E*polN[2], -E*complex(polN[0], polN[1])/np.sqrt(2)]

    # Populate the matrix
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];

            # Flip if needed
            if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
                sTemp = s2; s2 = s1; s1 = sTemp

            # Only apply if selection rules are satisfied
            if (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1) and (abs(s1.j - s2.j) <= 1):
                q = int((s2.mj - s1.mj))
                eff_E = 100*E_pol[1+q]
                H[x,y] = -atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,q)*(C_e*a0/C_h)*(eff_E)
            
                # Conjugate to other side
                H[y,x] = np.conjugate(H[x,y])
            
    return stateNum, H

def getH_E_DC_singleAtom_fromBase(H_E_DC_singleAtom_base, qMax):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet DC E-field Hamiltonian by copying the input
    two-atom DC E-field Hamiltonian 2*qMax+1 times.
    Inputs:
        H_E_DC_twoAtom_base: the output of getH_E_DC_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
    Output:
        H: the two-atom Floquet DC E-field Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    dim = H_E_DC_singleAtom_base.shape[0]
    H = sp.lil_matrix((dim*(2*qMax + 1), dim*(2*qMax + 1)), dtype='complex_')
    # Copy the base H into the diagonal blocks
    for n in np.arange(0, 2*qMax+1):
        H[n*dim:(n+1)*dim, n*dim:(n+1)*dim] = H_E_DC_singleAtom_base
    return H

def getH_E_DC_singleAtom(oneAtomStateList, soi, pol, qMax, E = 1):
    '''
    This method constructs the two-atom Floquet DC E-field Hamiltonian.
    It is the typical method a user should use to get the full Floquet
    DC E-field Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        pol: a 3-element Python list with information about the direction of the 
             electric field vector [x,y,z]; does not need to be normalized
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        E (optional): the magnitude of the field (V/cm); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
                 (but before diagonalizing)
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_E_DC_singleAtom_base(oneAtomStateList, soi, pol, E = E)
    return getH_E_DC_singleAtom_fromBase(H1, qMax)


###########################################################################
#
# AC Electric Field Hamiltonian Construction - Single Atom
#
###########################################################################


def getH_E_AC_singleAtom_base(oneAtomStateList, soi, pol, E = 1):
    '''
    [Primarily a utility method]
    Get the part of the one-atom Hamiltonian containing the AC electric field couplings.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        pol: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        E (optional): the magnitude of the field (V/m); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    
    stateNum = len(oneAtomStateList) # state number
    H = sp.lil_matrix((stateNum, stateNum), dtype='complex_')
    polN = pol/np.linalg.norm(pol)
    
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];

            # # Flip if needed
            # if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
            #     sTemp = s2; s2 = s1; s1 = sTemp

            # Only apply if selection rules are satisfied
            if (s1 != s2) and (abs(s1.j - s2.j) <= 1) and (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1):
                q = int((s2.mj - s1.mj))
                eff_E = E * polN[1+q]
                #H[x,y] = atom.getRabiFrequency2(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,q,eff_E)/(2*pi)/2
                H[x,y] = atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,s2.mj - s1.mj)*(C_e*a0/C_h)*eff_E/2
                H[y,x] = np.conjugate(H[x,y])
            
    return stateNum, H

def getH_E_AC_singleAtom_fromBase(H_E_AC_singleAtom_base, qMax):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet AC E-field Hamiltonian by copying the input
    two-atom AC E-field Hamiltonian onto the relevant 2*(2*qMax) off-diagonal blocks.
    Inputs:
        H_E_AC_twoAtom_base: the output of getH_E_AC_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
    Output:
        H: the two-atom Floquet DC E-field Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    dim = H_E_AC_singleAtom_base.shape[0]
    H = sp.lil_matrix((dim*(2*qMax + 1), dim*(2*qMax + 1)), dtype='complex_')
    for n in np.arange(0, 2*qMax+1-1):
        H[n*dim:(n+1)*dim, (n+1)*dim:(n+2)*dim] = H_E_AC_singleAtom_base
    H2 = H.copy() + H.copy().conjugate().transpose()
    return H2

def getH_E_AC_singleAtom(oneAtomStateList, soi, pol, qMax, E = 1):
    '''
    This method constructs the two-atom Floquet AC E-field Hamiltonian.
    It is the typical method a user should use to get the full Floquet
    AC E-field Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        pol: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        E (optional): the magnitude of the AC field (V/m); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
                 (but before diagonalizing)
             NB: One can convert to |E| from a W/m^2 intensity via
                 np.sqrt(2*rf_intensity/(C_c*epsilon0))
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_E_AC_singleAtom_base(oneAtomStateList, soi, pol, E = E)
    return getH_E_AC_singleAtom_fromBase(H1, qMax)


###########################################################################
#
# Diagonal Parts of the Hamiltonian - Single Atom
#
###########################################################################


def getH_diag_singleAtom_base(oneAtomStateList, soi, B):
    '''
    [Primarily a utility method]
    Get the diagonal components of the one-atom Hamiltonian, which include Zeeman shifts.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        B: the magnetic field, in Gauss, that defines the quantization axis
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    fsl = oneAtomStateList
    stateNum = len(oneAtomStateList) # Floquet state number
    H = sp.lil_matrix((stateNum, stateNum))
    zeemanRef = atom.getZeemanEnergyShift(soi.l, soi.j, soi.mj, B*1e-4)/C_h

    for x in range(stateNum):
        H[x,x] = atom.getTransitionFrequency(soi.n, soi.l, soi.j, fsl[x].n ,fsl[x].l, fsl[x].j) + (atom.getZeemanEnergyShift(fsl[x].l,fsl[x].j,fsl[x].mj,B*1e-4)/C_h) - zeemanRef + 1e-6
        
    return stateNum, H

def getH_diag_singleAtom_fromBase(H_diag_singleAtom_base, qMax, rf_freq):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet Hamiltonian diagonal by copying the input
    diagonl 2*qMax + 1 times with the appropriate addition of multiples of the Floquet frequency.
    Inputs:
        H_diag_twoAtom_base: the output of getH_diag_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        rf_freq: the Floquet frequency to work in (Hz)
    Output:
        H: the two-atom Floquet diagonal Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    H = sp.lil_matrix((H_diag_singleAtom_base.shape[0]*(2*qMax + 1), H_diag_singleAtom_base.shape[0]*(2*qMax + 1)))
    diagElements0 = np.array([H_diag_singleAtom_base.diagonal(0)])
    diagElements = np.array([diagElements0 + q*rf_freq for q in np.arange(-qMax, qMax+1, 1)]).flatten()
    H.setdiag(diagElements)
    return H

def getH_diag_singleAtom(oneAtomStateList, soi, B, qMax, rf_freq):
    '''
    This method constructs the two-atom Floquet Hamiltonian diagonal elements,
    including Zeeman shifts (treated as diagonal in this code). 
    It is the typical method a user should use to get the full Floquet
    diagonal portion of the Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        B: the B-field (in Gauss)
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        rf_freq: the Floquet frequency to work in (Hz)
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_diag_singleAtom_base(oneAtomStateList, soi, B)
    return getH_diag_singleAtom_fromBase(H1, qMax, rf_freq)



###########################################################################
#
# Getting Hamiltonians - Diagonal (plus Zeeman shifts)
#
###########################################################################

def getH_diag_oneAtom_base(oneAtomStateList, soi, B):
    '''
    [Primarily a utility method]
    Get the diagonal components of the one-atom Hamiltonian, which include Zeeman shifts.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        B: the magnetic field, in Gauss, that defines the quantization axis
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    fsl = oneAtomStateList
    stateNum = len(oneAtomStateList) # Floquet state number
    H = sp.lil_matrix((stateNum, stateNum))
    zeemanRef = atom.getZeemanEnergyShift(soi.l, soi.j, soi.mj, B*1e-4)/C_h

    for x in range(stateNum):
        H[x,x] = atom.getTransitionFrequency(soi.n, soi.l, soi.j, fsl[x].n ,fsl[x].l, fsl[x].j) + (atom.getZeemanEnergyShift(fsl[x].l,fsl[x].j,fsl[x].mj,B*1e-4)/C_h) - zeemanRef + 1e-6
        
    return stateNum, H

def getH_diag_twoAtom_base(H_diag_oneAtom_base):
    '''
    [Primarily a utility method]
    This method forms the diagonal portion of the two-atom Hamiltonian from the diagonal
    portion of the one-atom Hamiltonian by simply Kronecker summing. The returned
    matrix is a scipy.sparse.lil_matrix object.
    '''
    return sp.kronsum(H_diag_oneAtom_base, H_diag_oneAtom_base, format="lil")

def getH_diag_twoAtom_fromBase(H_diag_twoAtom_base, qMax, rf_freq):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet Hamiltonian diagonal by copying the input
    diagonl 2*qMax + 1 times with the appropriate addition of multiples of the Floquet frequency.
    Inputs:
        H_diag_twoAtom_base: the output of getH_diag_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        rf_freq: the Floquet frequency to work in (Hz)
    Output:
        H: the two-atom Floquet diagonal Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    H = sp.lil_matrix((H_diag_twoAtom_base.shape[0]*(2*qMax + 1), H_diag_twoAtom_base.shape[0]*(2*qMax + 1)))
    diagElements0 = np.array([H_diag_twoAtom_base.diagonal(0)])
    diagElements = np.array([diagElements0 + q*rf_freq for q in np.arange(-qMax, qMax+1, 1)]).flatten()
    H.setdiag(diagElements)
    return H

def getH_diag_twoAtom(oneAtomStateList, soi, B, qMax, rf_freq):
    '''
    This method constructs the two-atom Floquet Hamiltonian diagonal elements,
    including Zeeman shifts (treated as diagonal in this code). 
    It is the typical method a user should use to get the full Floquet
    diagonal portion of the Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        B: the B-field (in Gauss)
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        rf_freq: the Floquet frequency to work in (Hz)
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_diag_oneAtom_base(oneAtomStateList, soi, B)
    H2 = getH_diag_twoAtom_base(H1)
    return getH_diag_twoAtom_fromBase(H2, qMax, rf_freq)

###########################################################################
#
# Getting Hamiltonians - DC E-field
#
###########################################################################

def getH_E_DC_oneAtom_base(oneAtomStateList, soi, pol, E = 1):
    '''
    [Primarily a utility method]
    Get the part of the one-atom Hamiltonian containing the DC electric field couplings.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        pol: a 3-element Python list with information about the direction of the 
             electric field vector [x,y,z]; does not need to be normalized
        E (optional): the magnitude of the field (V/cm); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    stateNum = len(oneAtomStateList) # state number
    H = sp.lil_matrix((stateNum, stateNum), dtype='complex_')

    # Normalize the polarization vector, and convert it to the atomic spherical basis
    polN = pol/np.linalg.norm(pol)
    E_pol = [E*complex(polN[0], -polN[1])/np.sqrt(2), E*polN[2], -E*complex(polN[0], polN[1])/np.sqrt(2)]

    # Populate the matrix
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];

            # Flip if needed
            if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
                sTemp = s2; s2 = s1; s1 = sTemp

            # Only apply if selection rules are satisfied
            if (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1) and (abs(s1.j - s2.j) <= 1):
                q = int((s2.mj - s1.mj))
                eff_E = 100*E_pol[1+q]
                H[x,y] = -atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,q)*(C_e*a0/C_h)*(eff_E)
            
                # Conjugate to other side
                H[y,x] = np.conjugate(H[x,y])
            
    return stateNum, H

def getH_E_DC_twoAtom_base(H_E_DC_oneAtom):
    '''
    [Primarily a utility method]
    This method forms the DC E-field portion of the two-atom Hamiltonian from the
    DC E-field portion of the one-atom Hamiltonian by simply Kronecker summing. The
    returned matrix is a scipy.sparse.lil_matrix object.
    '''
    return sp.kronsum(H_E_DC_oneAtom, H_E_DC_oneAtom, format="lil")

def getH_E_DC_twoAtom_fromBase(H_E_DC_twoAtom_base, qMax):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet DC E-field Hamiltonian by copying the input
    two-atom DC E-field Hamiltonian 2*qMax+1 times.
    Inputs:
        H_E_DC_twoAtom_base: the output of getH_E_DC_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
    Output:
        H: the two-atom Floquet DC E-field Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    dim = H_E_DC_twoAtom_base.shape[0]
    H = sp.lil_matrix((dim*(2*qMax + 1), dim*(2*qMax + 1)), dtype='complex_')
    # Copy the base H into the diagonal blocks
    for n in np.arange(0, 2*qMax+1):
        H[n*dim:(n+1)*dim, n*dim:(n+1)*dim] = H_E_DC_twoAtom_base
    return H

def getH_E_DC_twoAtom(oneAtomStateList, soi, pol, qMax, E = 1):
    '''
    This method constructs the two-atom Floquet DC E-field Hamiltonian.
    It is the typical method a user should use to get the full Floquet
    DC E-field Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        pol: a 3-element Python list with information about the direction of the 
             electric field vector [x,y,z]; does not need to be normalized
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        E (optional): the magnitude of the field (V/cm); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
                 (but before diagonalizing)
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_E_DC_oneAtom_base(oneAtomStateList, soi, pol, E = E)
    H2 = getH_E_DC_twoAtom_base(H1)
    return getH_E_DC_twoAtom_fromBase(H2, qMax)

###########################################################################
#
# Getting Hamiltonians - AC E-field
#
###########################################################################

def getH_E_AC_oneAtom_base(oneAtomStateList, soi, pol, E = 1):
    '''
    [Primarily a utility method]
    Get the part of the one-atom Hamiltonian containing the AC electric field couplings.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        soi: a "state of interest", which will serve as the zero-energy reference
        pol: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        E (optional): the magnitude of the field (V/m); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
    Outputs:
        stateNum: the number of states involved (i.e. the dimension of the matrix)
        H: the Hamiltonian produced, as a scipy.sparse.lil_matrix object
    '''
    
    stateNum = len(oneAtomStateList) # state number
    H = sp.lil_matrix((stateNum, stateNum), dtype='complex_')
    polN = pol/np.linalg.norm(pol)
    
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];

            # # Flip if needed
            # if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
            #     sTemp = s2; s2 = s1; s1 = sTemp

            # Only apply if selection rules are satisfied
            if (s1 != s2) and (abs(s1.j - s2.j) <= 1) and (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1):
                q = int((s2.mj - s1.mj))
                eff_E = E * polN[1+q]
                #H[x,y] = atom.getRabiFrequency2(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,q,eff_E)/(2*pi)/2
                H[x,y] = atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,s2.mj - s1.mj)*(C_e*a0/C_h)*eff_E/2
                H[y,x] = np.conjugate(H[x,y])
            
    return stateNum, H

def getH_E_AC_twoAtom_base(H_E_AC_oneAtom):
    '''
    [Primarily a utility method]
    This method forms the AC E-field portion of the two-atom Hamiltonian from the
    AC E-field portion of the one-atom Hamiltonian by simply Kronecker summing. The
    returned matrix is a scipy.sparse.lil_matrix object.
    '''
    return sp.kronsum(H_E_AC_oneAtom, H_E_AC_oneAtom, format="lil")

def getH_E_AC_twoAtom_fromBase(H_E_AC_twoAtom_base, qMax):
    '''
    [Primarily a utility method]
    This method forms the full two-atom Floquet AC E-field Hamiltonian by copying the input
    two-atom AC E-field Hamiltonian onto the relevant 2*(2*qMax) off-diagonal blocks.
    Inputs:
        H_E_AC_twoAtom_base: the output of getH_E_AC_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
    Output:
        H: the two-atom Floquet DC E-field Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    dim = H_E_AC_twoAtom_base.shape[0]
    H = sp.lil_matrix((dim*(2*qMax + 1), dim*(2*qMax + 1)), dtype='complex_')
    for n in np.arange(0, 2*qMax+1-1):
        H[n*dim:(n+1)*dim, (n+1)*dim:(n+2)*dim] = H_E_AC_twoAtom_base
    H2 = H.copy() + H.copy().conjugate().transpose()
    return H2

def getH_E_AC_twoAtom(oneAtomStateList, soi, pol, qMax, E = 1):
    '''
    This method constructs the two-atom Floquet AC E-field Hamiltonian.
    It is the typical method a user should use to get the full Floquet
    AC E-field Hamiltonian.
    Inputs:
        oneAtomStateList: the Python list of oneAtomState objects to serve as the
                          one-atom basis
        soi: the "state of interest", which will be the zero-energy reference
        pol: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
        E (optional): the magnitude of the AC field (V/m); default 1
             NB: You can always multiply the resulting Hamiltonian by |E| later
                 (but before diagonalizing)
             NB: One can convert to |E| from a W/m^2 intensity via
                 np.sqrt(2*rf_intensity/(C_c*epsilon0))
    Output:
        H: the two-atom Floquet Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    _, H1 = getH_E_AC_oneAtom_base(oneAtomStateList, soi, pol, E = E)
    H2 = getH_E_AC_twoAtom_base(H1)
    return getH_E_AC_twoAtom_fromBase(H2, qMax)

###########################################################################
#
# Getting Hamiltonians - Dipole-dipole interactions
#
###########################################################################

def getH_V_dd_oneAtom_base(oneAtomStateList):
    '''
    [Primarily a utility method]
    This method constructs a matrix of the dipole elements between one-atom states,
    to be used judiciously to create the two-atom dipole-dipole matrix.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
    Outputs:
        MDref: the reference dipole matrix produced, as a scipy.sparse.lil_matrix object
    '''
    stateNum = len(oneAtomStateList) # state number

    MDref = sp.lil_matrix((stateNum, stateNum), dtype='complex_')
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];
            # flipMult = 1
            # # Flip if needed
            # if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
            #     sTemp = s2; s2 = s1; s1 = sTemp
            #     # Implement the rule that T(k)_q^\dagger = (-1)^q T(k)_{-q}
            #     flipMult = (-1)**(s2.mj-s1.mj)
            # Only apply if selection rules are satisfied
            if (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1) and (abs(s1.j - s2.j) <= 1):
                # sn = np.sign(atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j))
                MDref[x,y] = atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,s2.mj - s1.mj)*(C_e*a0)
                MDref[y,x] = np.conjugate(MDref[x,y])*(-1)**(s2.mj-s1.mj)

    return MDref

def getH_V_dd_twoAtom_base(oneAtomStateList, theta, phi, MDref = []):
    '''
    This method gets the full two-atom dipole-dipole Hamiltonian, except for the constant
    1/r^3 factor. This function implements a form of Vdd found in Deniz's candidacy paper.
    Inputs:
        oneAtomStateList: a Python list of oneAtomState objects to be used as the basis
        theta, phi: the polar and azimuthal angles of the interatomic axis between the atoms,
                    referenced to the quantization axis
        MDref (optional): if given, use this as the one-atom reference dipole element matrix;
                          default []
    Outputs:
        twoAtomSL: the two-atom state list getTwoAtomFloquetStateList(oneAtomStateList, 0)
        Hdd: the two-atom dipole-dipole Hamiltonian, as a scipy.sparse.lil_matrix object,
             except for a 1/r^3 factor to be multiplied in externally
    '''
    if (len(MDref) == 0):
        # If no MDref given, have to construct it
        MDref = getH_V_dd_oneAtom_base(oneAtomStateList)

    # Clebsch-Gordan map
    CGmap = {(1,1): 1, (1,0): 1/np.sqrt(2), (0,1): 1/np.sqrt(2), (-1,1): 1/np.sqrt(6), (1,-1): 1/np.sqrt(6), (0,0): np.sqrt(2/3), (-1,-1): 1, (-1,0): 1/np.sqrt(2), (0,-1): 1/np.sqrt(2)}
    # Spherical harmonic values map
    y2sph = {-2: (1/4)*np.sqrt(15/(2*pi))*np.sin(theta)**2*np.exp(2j*phi),
          -1: (1/2)*np.sqrt(15/(2*pi))*np.sin(theta)*np.cos(theta)*np.exp(1j*phi),
          0: (1/4)*np.sqrt(5/(pi))*(3*np.cos(theta)**2 - 1),
          1: -(1/2)*np.sqrt(15/(2*pi))*np.sin(theta)*np.cos(theta)*np.exp(-1j*phi),
          2: (1/4)*np.sqrt(15/(2*pi))*np.sin(theta)**2*np.exp(-2j*phi)
         }
    
    stateNum = len(oneAtomStateList) # state number
    twoAtomSL = getTwoAtomFloquetStateList(oneAtomStateList, 0)
    Hdd = sp.lil_matrix((stateNum**2, stateNum**2), dtype='complex_')

    # Populate the matrix
    for x in range(stateNum**2):
        for y in range(x+1, stateNum**2):
            s1 = twoAtomSL[x]; s2 = twoAtomSL[y];
            # Only apply if selection rules are satisfied for both atoms
            if (abs(s1.mj1 - s2.mj1) <= 1) and (abs(s1.l1 - s2.l1) == 1) and (abs(s1.j1 - s2.j1) <= 1) and (abs(s1.mj2 - s2.mj2) <= 1) and (abs(s1.l2 - s2.l2) == 1) and (abs(s1.j2 - s2.j2) <= 1):
                # A two-atom state corresponds to two one-atom states... these algorithms
                # generate very well-ordered one-atom states, so we can just use indices
                # to find which one-atom states are in our two-atom state
                x1, x2 = divmod(x, stateNum); y1, y2 = divmod(y, stateNum)
                # Then get the two dipole matrix elements
                d1 = MDref[x1, y1]; d2 = MDref[x2, y2]
                q1 = s2.mj1 - s1.mj1; q2 = s2.mj2 - s1.mj2;
                CG = CGmap[(q1,q2)]
                # Put everything together
                Hdd[x,y] = -np.sqrt(24*pi/5)*(1/(4*pi*epsilon0)/C_h)*CG*d1*d2*y2sph[q1+q2]
    Hdd = Hdd + Hdd.copy().transpose().conjugate()

    return twoAtomSL, Hdd

def getH_dd_twoAtom_fromBase(H_dd_twoAtom_base, qMax):
    '''
    This method constructs the two-atom Floquet dipole-dipole Hamiltonian.
    It is the typical method a user should use to get the full Floquet
    dipole-dipole portion of the Hamiltonian.
    Inputs:
        H_dd_twoAtom_base: the output of getH_V_dd_twoAtom_base()
        qMax: the maximum Fourier component to use (so that -qMax <= q <= qMax)
    Output:
        H: the two-atom Floquet dipole-dipole Hamiltonian, as a scipy.sparse.lil_matrix
    '''
    dim = H_dd_twoAtom_base.shape[0]
    H = sp.lil_matrix((dim*(2*qMax + 1), dim*(2*qMax + 1)), dtype='complex_')
    # Copy the input into 2*qMax+1 blocks on the diagonal
    for n in np.arange(0, 2*qMax+1):
        H[n*dim:(n+1)*dim, n*dim:(n+1)*dim] = H_dd_twoAtom_base
    return H




###########################################################################
#
# Function for sweeping AC Frequency with single atoms
#
###########################################################################


def sweepACfreqSingleAtom(low_n, high_n, max_l, qMax, soi, b_field, e_field, rf_freqs, rf_inten, dc_polarization, ac_polarization, stateInds=[], keepEVNum = 0, saveFile = "", progress_notif = 10, stateList=0, verbose=True):
    '''
    This function does the work of ramping the AC/Floquet frequency and, at each point, 
    diagonalizing the Hamiltonian. Use with caution as it can take a lot of time and also
    max out your computer's available RAM.
    NB: This function also works just fine for diagonalizing at a single RF/Floquet frequency;
        just pass in a one-item rf_freqs list.

    Inputs: 
        low_n, high_n, max_l: parameters to determine the one-atom states to be used. States
            will be selected such that low_n <= n <= high_n (principal quantum number) and
            0 <= l <= max_l.
        qMax: the max Fourier component to use for the Floquet basis, such that -qMax <= q <= qMax.
        soi: the zero-energy reference state, as a oneAtomState object. (If the state given is |s1>,
             the zero-energy state in the two-atom basis willbe |s1s1>.
        b_field: the magnetic field magnitude (Gauss)
        e_field: the magnitude of the field (V/cm)
        rf_freqs: a NumPy array of RF frequencies to diagonalize across (Hz)
        rf_inten: the RF intensity (W/m^2)
        dc_polarization: a 3-element Python list with information about the direction of the 
             DC electric field vector [x,y,z]; does not need to be normalized
        ac_polarization: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        stateInds (optional): if you want to keep only some of the states prior to diagonalization,
                  this argument should be an array of indices of the states to keep; if an
                  empty list [], keep all states (default [])
        keepEVNum (optional): the number of eigenvectors to calculate and return. If 0, uses NumPy
                diagonalization to get all of them; if greater than zero, uses SciPy numerical
                diagonalization to get the smallest-magnitude keepEVNum of them (default 0)
        saveFile (optional): the address of the file to save the data to; if an empty string (""),
                 don't save the data (default ""). If the given file already exists, the function
                 will check whether you actually want to overwrite that file before continuing, but
                 will overwrite the file if you say so (and will stop execution by throwing an error
                 if you say not to).
        progress_notif (optional): if verbose is True, will print out a progress update according
                       to this parameter. If -1, prints out a notification after every RF frequency
                       is handled; if progress_notif = n > 0, prints out a notification after
                       every n RF frequencies are handled (default 10).
        stateList (optional): while this function normally creates the one-atom state list automatically
                              based on the low_n, high_n, and max_l parameters, you can override that
                              behavior by passing in a Python list of oneAtomState objects using this
                              argument. If an empty list [], no override (default []).
        verbose (optional): If True, prints out some progress updates; if False, doesn't (default True).

    Outputs:
        oneAtomBareStates: the Python list of oneAtomState objects forming the basis (will
                           equal stateList if that argument is used ot override the usual behavior)
        eigenvalues: a dictionary of eigenvalues returned at each RF frequency, with keys 0, 1, 2, ...
        eigenvectors: a dictionary of eigenvectors returned at each RF frequency, with keys 0, 1, 2, ...
                      NB: The eigenvector array at each key is arranged such that each ROW is an eigenvector
                      (e.g. eigenvectors[0][0] is the eigenvector corresponding to the smallest-magnitude
                      eigenvalue). This is the TRANSPOSE of what np.linalg.eigh() and scipy.sparse.eigs()
                      return normally.
        Htot: the total Hamiltonian of the LAST RF frequency, in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
        H_DC: the DC Hamiltonian (same for all RF frequencies), in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
        H_AC: the AC Hamiltonian (same for all RF frequencies), in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
    
    '''

    # Do this check at the start to save work: is the specified file already a file?
    # If so, get input from the user on whether to override.
    f = None
    if (saveFile != ""):
        if (os.path.isfile(saveFile)):
            ans = input("The file you specified already exists. Do you want to overwrite it? [y]/n")
            if (ans == "n"):
                raise Exception("User stopped execution.")
            else:
                f = h5.File(saveFile, "w")
        else:
            f = h5.File(saveFile, "w")

    # Make the state list.
    oneAtomBareStates = []
    if (stateList != 0):
        oneAtomBareStates = stateList
    else:
        oneAtomBareStates = getStateList(low_n,high_n,max_l)
    dimH = len(oneAtomBareStates) * (2*qMax + 1)
    if verbose:
        print("Made the state lists. Total of {:.0f} states.".format(dimH), end=" ")

    # Get the DC Hamiltonian.
    H_DC = sp.lil_matrix((dimH, dimH))

    # If e_field = 0, no need to compute
    if (e_field != 0):
        H_DC = getH_E_DC_singleAtom(oneAtomBareStates, soi, dc_polarization, qMax, E = e_field)
        if verbose:
            print("Made DC Hamiltonian.", end=" ")
    # Get the AC Hamiltonian.
    H_AC = getH_E_AC_singleAtom(oneAtomBareStates, soi, ac_polarization, qMax, E = np.sqrt(2*rf_inten/(C_c*epsilon0)))
    if verbose:
        print("Made AC Hamiltonian.")

    # Get the *base* diagonal matrix... this has to be modified for every RF frequency,
    # but we can do a lot of the base computation in advance
    _, H_diag_singleAtom_base = getH_diag_singleAtom_base(oneAtomBareStates, soi, b_field)

    eigenvalues = {}
    eigenvectors = {}
    if verbose:
        print("Starting analysis of the Hamiltonians...")
    # Start the sweep.
    for r, rf_freq in enumerate(rf_freqs):
        # Get the diagonals for the RF frequency
        H_diag = getH_diag_singleAtom_fromBase(H_diag_singleAtom_base, qMax, rf_freq)
        # Total Hamiltonian
        Htot = (H_diag + H_DC + H_AC).toarray()
        if (len(stateInds) != 0):
            Htot = Htot.copy()[stateInds][:,stateInds]

        # Get the eigenvalues and eigenvectors
        if (keepEVNum == 0):
            vals, vecs = np.linalg.eigh(Htot)
            eigenvalues[r] = vals
            eigenvectors[r] = np.transpose(vecs)
        else:
            Hinv = sp.lil_matrix(np.linalg.inv(Htot))
            vals, vecs = sp.linalg.eigs(Hinv, k=keepEVNum, which='LM')
            eigenvalues[r] = 1/vals
            eigenvectors[r] = np.transpose(vecs)

        # Save the data
        if (saveFile != ""):
            dset = f.create_dataset(str(r), data = eigenvectors[r])
            dset.attrs["n0"] = low_n; dset.attrs["n1"] = high_n; dset.attrs["max_l"] = max_l; dset.attrs["max_q"] = qMax;
            dset.attrs["soi"] = soi.tolist(); dset.attrs["si"] = stateInds;
            dset.attrs["B"] = b_field;
            dset.attrs["E"] = e_field; dset.attrs["E_DC_pol"] = dc_polarization;
            dset.attrs["rf_freq"] = rf_freq; dset.attrs["rf_inten"] = rf_inten; dset.attrs["E_AC_pol"] = ac_polarization;
            dset.attrs["theta"] = theta; dset.attrs["phi"] = phi; dset.attrs["DDmult"] = DDmult;
            dset.attrs["evals"] = np.real(eigenvalues[r]);

        # Update on progress
        if (verbose and (progress_notif == -1) or ((r+1)%progress_notif == 0)):
            print("{:.0f} done...".format(r+1), end=" ")

    # Make sure to close the file properly
    if (saveFile != ""):
        f.close()
    
    return oneAtomBareStates, eigenvalues, eigenvectors, Htot, H_DC, H_AC


def sweepACfreqSingleAtom_plot(states, qMax, rf_freqs, plotLbls, cmaps, stateList, stateInds, evals, evecs, ylim=[], title="", s0 = 10, locArg='best', plotAll = True, savefig=""):
    '''
    This function plots the results of an AC frequency sweep as calculated by sweepACfreq().
    It will color in an arbitrary number of state overlaps with the colormaps given, and can
    save the figure if desired.

    Inputs:
        The first two parameters are used together to determine the states to plot colored
        overlaps of; see getFracsOfState()'s soi and SAS parameters for details.
        
        states: a list of oneAtomState objects
        SASs: a list of integers
        
        qMax: the max Fourier component used to generate the eigenvalues being plotted
              (so this function can reconstruct the basis used in that calculation)
        rf_freqs: the RF frequencies used in the calculation to be plotted (Hz)
                  (note that they will be plotted in MHz)

        The next two parameters are for plotting.

        plotLbls: the legend labels corresponding to the states chosen for overlap coloring
                  (list of strings).
        cmaps: the colormaps corresponding to the states chosen for overlap coloring
               (list of matplotlib.colors.ListedColormap objects).

        The next four parameters are the data to plot.

        stateList: a Python list of oneAtomState objects that were the one-atom
                          basis used when generating the input eigenvectors (so this
                          function can reconstruct the basis that those eigenvectors are
                          expressed in)
        stateInds: a list of indices of states to keep in the two-atom Floquet basis; if
                   keeping all of the states, use the empty set []
        evals: the dictionary of eigenvalues to be plotted, with keys 0, 1, 2, ..., as
               returned by the functions of this library
        evecs: the dictionary of eigenvectors to be plotted, with keys 0, 1, 2, ..., as
               returned by the functions of this library
        ylim (optional): a two-item list for manually specifying y-axis limits; if you want to
                         set these such that all the eigenvalues at all frequencies are shown,
                         use [] (default [])
        title (optional): the title of the plot, e.g. for showing parameters used (default "")
        s0 (optional): a marker size for axes.scatter() (default 10)
        locArg (optional): a location argument for the legend; passed directly to ax.legend(),
                           so all options available there are available here (default 'best')
        plotAll (optional): if True, in addition to all of the colored eigenvalues, will plot
                            ALL of the given eigenvalues in a light gray behind the colored
                            eigenvales (default True)
        savefig (optional): a file address to save the figure; no save if an empty string ""
                            (default "")

    Outputs: None (shows a plot)
    '''

    # For generating the legend later)
    plotCircs = []
    for j in range(len(plotLbls)):
        plotCircs.append(mlines.Line2D([], [], color=cmaps[j](256), marker='o', linestyle='None', markersize=8, label=plotLbls[j]))

    # Do the state overlap calculation and plotting
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(len(rf_freqs)):
        evNum = len(evals[i])
        if plotAll:
            sc0 = ax.scatter(np.array([rf_freqs[i]]*evNum)/1e6, evals[i]/1e6, c=[[0,0,0,0.1]], s=s0)
        for j in range(len(plotLbls)):
            colFracs = getFracsOfStateSingleAtom(stateList, qMax, evecs[i], states[j], stateInds=stateInds)
            sc = ax.scatter(np.array([rf_freqs[i]]*evNum)/1e6, evals[i]/1e6, c=colFracs, cmap=cmaps[j], s=s0, vmin=0, vmax=1)

    # Set plot properties
    ax.set_xlabel("Microwave Frequency (MHz)")
    ax.set_ylabel("Eigenenergy (MHz)")
    if (len(ylim) != 0):
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ax.set_ylim(min(evals[0][0], evals[int(len(rf_freqs)-1)][0])/1e6*1.05, max(evals[0][-1], evals[int(len(rf_freqs)-1)][-1])/1e6*1.05)
    if (title != ""):
        ax.set_title(title)

    # The extras: the legend, the colorbar, saving, etc
    ax.legend(handles=plotCircs, loc=locArg)
    fig.colorbar(sc, ax=ax, pad=0.00)
    fig.tight_layout()
    if (savefig != ""):
        plt.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True)
    plt.show()



###########################################################################
#
# Functions for sweeping the relative atom position
#
###########################################################################

def sweepPosition(low_n, high_n, max_l, qMax, soi, b_field, e_field, rf_freq, rf_inten, dc_polarization, ac_polarization, theta, phi, rs, rf_phase=0,  stateInds=[], keepEVNum = 0, saveFile = "", progress_notif = 10, stateList=0, verbose=True):
    '''
    This function does the work of ramping the AC/Floquet frequency and, at each point, 
    diagonalizing the Hamiltonian. Use with caution as it can take a lot of time and also
    max out your computer's available RAM.
    NB: This function also works just fine for diagonalizing at a single RF/Floquet frequency;
        just pass in a one-item rf_freqs list.

    Inputs: 
        low_n, high_n, max_l: parameters to determine the one-atom states to be used. States
            will be selected such that low_n <= n <= high_n (principal quantum number) and
            0 <= l <= max_l.
        qMax: the max Fourier component to use for the Floquet basis, such that -qMax <= q <= qMax.
        soi: the zero-energy reference state, as a oneAtomState object. (If the state given is |s1>,
             the zero-energy state in the two-atom basis willbe |s1s1>.
        b_field: the magnetic field magnitude (Gauss)
        e_field: the magnitude of the field (V/cm)
        rf_freq: the RF/Floquet frequency to diagonalize with (Hz)
        rf_inten: the RF intensity (W/m^2)
        dc_polarization: a 3-element Python list with information about the direction of the 
             DC electric field vector [x,y,z]; does not need to be normalized
        ac_polarization: a 3-element Python list with information about the polarization of the 
             AC field, [sigma-, pi, sigma+]; does not need to be normalized
        theta, phi: the polar and azimuthal angles of the interatomic axis between the atoms,
                    referenced to the quantization axis
        rs: a NumPy array of two-atom radii to diagonalize for (in m)
        rf_phase (optional): the phase of the RF field (in radians) (default 0)
        stateInds (optional): if you want to keep only some of the states prior to diagonalization,
                  this argument should be an array of indices of the states to keep; if an
                  empty list [], keep all states (default [])
        keepEVNum (optional): the number of eigenvectors to calculate and return. If 0, uses NumPy
                diagonalization to get all of them; if greater than zero, uses SciPy numerical
                diagonalization to get the smallest-magnitude keepEVNum of them (default 0)
        saveFile (optional): the address of the file to save the data to; if an empty string (""),
                 don't save the data (default ""). If the given file already exists, the function
                 will check whether you actually want to overwrite that file before continuing, but
                 will overwrite the file if you say so (and will stop execution by throwing an error
                 if you say not to).
        progress_notif (optional): if verbose is True, will print out a progress update according
                       to this parameter. If -1, prints out a notification after every RF frequency
                       is handled; if progress_notif = n > 0, prints out a notification after
                       every n RF frequencies are handled (default 10).
        stateList (optional): while this function normally creates the one-atom state list automatically
                              based on the low_n, high_n, and max_l parameters, you can override that
                              behavior by passing in a Python list of oneAtomState objects using this
                              argument. If an empty list [], no override (default []).
        verbose (optional): If True, prints out some progress updates; if False, doesn't (default True).

    Outputs:
        oneAtomBareStates: the Python list of oneAtomState objects forming the basis (will
                           equal stateList if that argument is used ot override the usual behavior)
        eigenvalues: a dictionary of eigenvalues returned at each RF frequency, with keys 0, 1, 2, ...
        eigenvectors: a dictionary of eigenvectors returned at each RF frequency, with keys 0, 1, 2, ...
                      NB: The eigenvector array at each key is arranged such that each ROW is an eigenvector
                      (e.g. eigenvectors[0][0] is the eigenvector corresponding to the smallest-magnitude
                      eigenvalue). This is the TRANSPOSE of what np.linalg.eigh() and scipy.sparse.eigs()
                      return normally.
        Htot: the total Hamiltonian of the LAST RF frequency, in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
        H_DC: the DC Hamiltonian (same for all RF frequencies), in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
        H_AC: the AC Hamiltonian (same for all RF frequencies), in the two-atom Floquet basis,
              as a scipy.sparse.lil_matrix
        Hdd_base: the output of getH_V_dd_twoAtom_base() (same for all RF frequencies), which
                  is in the two-atom but NOT the Floquet basis, as a scipy.sparse.lil_matrix
    
    '''
    
    # Do this check at the start to save work: is the specified file already a file?
    # If so, get input from the user on whether to override.
    f = None
    if (saveFile != ""):
        if (os.path.isfile(saveFile)):
            ans = input("The file you specified already exists. Do you want to overwrite it? [y]/n")
            if (ans == "n"):
                raise Exception("User stopped execution.")
            else:
                f = h5.File(saveFile, "w")
        else:
            f = h5.File(saveFile, "w")
    
    # Make the state list.
    oneAtomBareStates = []
    if (stateList != 0):
        oneAtomBareStates = stateList
    else:
        oneAtomBareStates = getStateList(low_n,high_n,max_l)
    dimH = len(oneAtomBareStates)**2 * (2*qMax + 1)
    if verbose:
        print("Made the state lists. Total of {:.0f} states.".format(dimH), end=" ")

    # Get the diagonal Hamiltonian.
    H_diag = getH_diag_twoAtom(oneAtomBareStates, soi, b_field, qMax, rf_freq)
    if verbose:
        print("Made diagonal Hamiltonian.")
    # Get the DC Hamiltonian.
    H_DC = sp.lil_matrix((dimH, dimH))
    # If e_field = 0, no need to compute
    if (e_field != 0):
        int_H_DC = getH_E_DC_twoAtom(oneAtomBareStates, soi, dc_polarization, qMax, E = e_field)
        if verbose:
            print("Made DC Hamiltonian.", end=" ")
    # Get the AC Hamiltonian.
    H_AC = getH_E_AC_twoAtom(oneAtomBareStates, soi, ac_polarization, qMax, E = np.sqrt(2*rf_inten/(C_c*epsilon0))*np.exp(1j*rf_phase))
    if verbose:
        print("Made AC Hamiltonian.")

    # Finally get the base of Hdd in advance...
    _, Hdd_base = getH_V_dd_twoAtom_base(oneAtomBareStates, theta, phi, MDref = [])

    eigenvalues = {}
    eigenvectors = {}
    if verbose:
        print("Starting analysis of the Hamiltonians...")
    # Start the sweep.
    for r, r0 in enumerate(rs):
        Hdd = (1/r0**3)*getH_dd_twoAtom_fromBase(Hdd_base, qMax)
        # Total Hamiltonian
        Htot = (H_diag + H_DC + H_AC + Hdd).toarray()
        if (len(stateInds) != 0):
            Htot = Htot.copy()[stateInds][:,stateInds]

        # Get the eigenvalues and eigenvectors
        if (keepEVNum == 0):
            vals, vecs = np.linalg.eigh(Htot)
            eigenvalues[r] = vals
            eigenvectors[r] = np.transpose(vecs)
        else:
            Hinv = sp.lil_matrix(np.linalg.inv(Htot))
            vals, vecs = sp.linalg.eigs(Hinv, k=keepEVNum, which='LM')
            eigenvalues[r] = 1/vals
            eigenvectors[r] = np.transpose(vecs)

        # Save the data
        if (saveFile != ""):
            dset = f.create_dataset(str(r), data = eigenvectors[r])
            dset.attrs["n0"] = low_n; dset.attrs["n1"] = high_n; dset.attrs["max_l"] = max_l; dset.attrs["max_q"] = qMax;
            dset.attrs["soi"] = soi.tolist(); dset.attrs["si"] = stateInds;
            dset.attrs["B"] = b_field;
            dset.attrs["E"] = e_field; dset.attrs["E_DC_pol"] = dc_polarization;
            dset.attrs["rf_freq"] = rf_freq; dset.attrs["rf_inten"] = rf_inten; dset.attrs["E_AC_pol"] = ac_polarization;
            dset.attrs["theta"] = theta; dset.attrs["phi"] = phi; dset.attrs["DDmult"] = (1/r0**3);
            dset.attrs["evals"] = np.real(eigenvalues[r]);

        # Update on progress
        if (verbose and (progress_notif == -1) or ((r+1)%progress_notif == 0)):
            print("{:.0f} done...".format(r+1), end=" ")

    # Make sure to close the file properly
    if (saveFile != ""):
        f.close()
    
    return oneAtomBareStates, eigenvalues, eigenvectors, Htot, H_DC, H_AC, Hdd_base


###########################################################################
#
# Data Loading and Fitting
#
###########################################################################

def loadRsweepData(filename):
    '''
    This function loads all of the data from a file containing the eigenvalues and eigenvectors
    for all data generated from a sweep of the position. This function pairs with
    the data storage of sweepPosition().

    Inputs: filename, the address of the file to be opened and read out

    Outputs: a tuple containing the following, in this order:
        low_n, high_n, max_l, qMax, state of interest (as a oneAtomState object),
        stateInds (the indices of the states that were kept), b_field, (DC) e_field,
        (DC) E-field polarization list, RF/Floquet frequency , RF intensity,
        (AC) E-field polarization list, theta, phi, list of rs (in m),
        eigenvalues (dictionary with keys 0, 1, 2, ...), eigenvectors
        (dictionary with keys 0, 1, 2, ...)
    '''
    
    f = h5.File(filename, "r")
    # Many of the parameters will be the same for all of the datasets in this file,
    # so get the first one now
    dset0 = f["0"]
    #keys = np.sort(np.array(list(f.keys()), dtype=int))
    sortedInds = np.argsort(np.array(list(f.keys()), dtype=float))
    keys = np.array(list(f.keys()))[sortedInds]
    # Loop through the keys and get the data
    rs = np.zeros(len(keys))
    evals = {}
    evecs = {}
    for k, key in enumerate(keys):
        dset = f[key]
        rs[k] = (1/dset.attrs["DDmult"])**(1/3)
        evals[k] = dset.attrs["evals"]
        evecsK = np.zeros(dset.shape, dtype='complex128')
        dset.read_direct(evecsK)
        evecs[k] = evecsK
        del evecsK, dset
        
    retVals = (dset0.attrs["n0"], dset0.attrs["n1"], dset0.attrs["max_l"], dset0.attrs["max_q"], oneAtomState(*dset0.attrs["soi"]), dset0.attrs["si"], dset0.attrs["B"], dset0.attrs["E"], dset0.attrs["E_DC_pol"], dset0.attrs["rf_freq"], dset0.attrs["rf_inten"], dset0.attrs["E_AC_pol"], dset0.attrs["theta"], dset0.attrs["phi"], rs, evals, evecs)
    f.close()
    return retVals


    repeated_list = []
    for _ in range(n):
        inner_list = []
        for item in original_list:
            inner_list.append(item)
        repeated_list.append(inner_list)
    return repeated_list

def AnalyzeImportedSweepData(dfs,stateIndices,twoAtomFSLlist,SOIList):
    rLenVec = np.zeros(len(dfs))
    for fileInd in range(len(dfs)):
        filename = dfs[fileInd]
        f = h5.File(filename, "r")
        keys = np.array(list(f.keys()))
        rLenVec[fileInd] = len(keys)

    stateLenVec = np.zeros(len(dfs))
    for i in range(len(stateIndices)):
        stateLenVec[i] = len(stateIndices[i])

    Es = np.zeros((len(dfs), int(max(rLenVec)), int(max(stateLenVec))))
    rsArr = np.zeros((len(dfs), int(max(rLenVec)), int(max(stateLenVec))))
    eigenstateInds = np.zeros((len(dfs), int(max(rLenVec)), int(max(stateLenVec))), dtype=int)
    fosArr = np.zeros((len(dfs), int(max(rLenVec)), int(max(stateLenVec))))
    # Optional additional array for fitting
    # Indices: {file index, state number index, fit parameter [C3, C6, offset]}
    fitParams = np.zeros((len(dfs), int(max(stateLenVec)), 3))


    for fileInd in range(len(dfs)):
        low_n, high_n, max_l, qMax, _, si, _, _, _, _, _, _, _, _, rs, evals, evecs = loadRsweepData(dfs[fileInd])

        if len(twoAtomFSLlist)==1:
            twoAtomFSL = twoAtomFSLlist[0]
        else:
            twoAtomFSL = twoAtomFSLlist[fileInd]

        if len(SOIList)==1:
            SOI = SOIList[0]
        else:
            SOI = SOIList[fileInd]

        if len(stateIndices)==1:
            stateIndicesTemp = stateIndices[0]
        else:
            stateIndicesTemp = stateIndices[fileInd]     


        soiInd = np.where(twoAtomFSL == SOI)[0][0]

        for rInd in range(int(rLenVec[fileInd])-1,-1,-1):
            ssVals = np.zeros(len(evecs[rInd]),dtype=np.complex_)
            for evecInd in range(len(evecs[rInd])):
                ssVals[evecInd] = evecs[rInd][evecInd][soiInd]

                sortedFOSInds = np.flip(np.argsort(np.abs(ssVals)))
                # If we're at the largest distance, we don't have a prior
                # state reference, but that's okay, we just pick the eigenstates
                # with the largest SOI components
                if (rInd == len(rs)-1):
                    for s in np.arange(len(stateIndicesTemp)):
                        # For each of the states we want...
                        eind = sortedFOSInds[stateIndicesTemp[s]] # That's our eigenstate
                        ee = np.real(evals[rInd][eind])/1e6 # There's the energy
                        # Store the data
                        eigenstateInds[fileInd,rInd,s] = eind
                        Es[fileInd,rInd,s] = ee
                        rsArr[fileInd,rInd,s] = rs[rInd]
                        fosArr[fileInd,rInd,s] = abs(ssVals)[eind]
                else:
                    for s in np.arange(len(stateIndicesTemp)):
                        # The algorithm is to find the eigenstate at this r value
                        # that has the largest overlap with the correspondingly
                        # indexed eigenstate at the previous r value. 
                        prevEigenstate = evecs[rInd+1][eigenstateInds[fileInd,rInd+1,s]]
                        overlaps = np.abs(np.dot(prevEigenstate, np.transpose(evecs[rInd])))**2
                        eind = np.argmax(overlaps)
                        ee = np.real(evals[rInd][eind])/1e6 # There's the energy
                        # Store the data
                        eigenstateInds[fileInd,rInd,s] = eind
                        Es[fileInd,rInd,s] = ee
                        rsArr[fileInd,rInd,s] = rs[rInd]
                        fosArr[fileInd,rInd,s] = abs(ssVals)[eind]

                for s in np.arange(len(stateIndicesTemp)):
                    popt, pcov = opt.curve_fit(lambda x, a, b, c: a/x**3 + b/x**6 + c, rsArr[fileInd,:-1,s], Es[fileInd,:-1,s], p0=np.array([-1e-14, 2e-28, 0.5]), maxfev=20000)
                    fitParams[fileInd,s,:] = popt


    return Es, rsArr, eigenstateInds, fosArr, fitParams


        
def plotAnalyzedSweepData(Es, rsArr, fitParams, stateIndices, dfs, Eb, refBool,ymin, ymax):
    colsBase = generate_shades([0,0,1],len(dfs),target_color=(0.7,0,0))

    stateLenVec = np.zeros(len(dfs))
    for i in range(len(stateIndices)):
        stateLenVec[i] = len(stateIndices[i])


    cols = np.zeros((len(dfs), int(max(stateLenVec)), len(rsArr[0])-1, 4))
    for j in range(len(dfs)):
        if len(stateIndices) == 1:
            stateIndicesTemp = stateIndices[0]
        else:
            stateIndicesTemp = stateIndices[j]
        for s in range(len(stateIndicesTemp)):
            cols[j,s,:,0] = colsBase[j][0];cols[j,s,:,1] = colsBase[j][1];cols[j,s,:,2] = colsBase[j][2]
            # This is opacity; if you want no opacity changes just set this to 1
            # cols[j,s,:,3] = fosArr[j,:,s][:-1]
            cols[j,s,:,3] = 1

    # This is a filled in array of r values for plotting
    # fit lines, shaded boundaries, etc.
    contRs = np.linspace(rsArr[0,0,0],rsArr[0,-2,0],int((rsArr[0,-2,0]-rsArr[0,0,0])*1e6*2+1))



    # Make the plot
    fig, ax = plt.subplots(figsize=(12,8))
    for j in range(len(dfs)):
        if len(stateIndices) == 1:
            stateIndicesTemp = stateIndices[0]
        else:
            stateIndicesTemp = stateIndices[j]
        for s in range(len(stateIndicesTemp)):
            # Set the reference point
            ref = refBool*Es[j,-1,s]
            # Plot the eigenvalues
            ax.scatter(rsArr[j,:-1,s]*1e6, Es[j,:-1,s]-ref, color=cols[j,s], marker='o', zorder=25, linestyle='-', linewidth=1)
            # Plot the blockade bands if desired
            ax.fill_between(contRs*1e6, [Es[j,-1,s]-ref-Eb]*len(contRs), [Es[j,-1,s]-ref+Eb]*len(contRs), color=colsBase[j], alpha=(1-refBool)*0.1 + (refBool)*0.05)
            # ax.plot(contRs*1e6, [Es[j,-1,s]-ref]*len(contRs), linestyle='--', color=colsBase[j], alpha=(1-refBool)*0.2 + (refBool)*0.1)
            # Plot the fit lines if desired
            ax.plot(contRs*1e6, fitParams[j,s,0]/contRs**3 + fitParams[j,s,1]/contRs**6 + fitParams[j,s,2]-ref,
                    color = colsBase[j], linestyle='dashed', zorder=20)
            # Plot lines between eigenvalues if desired
            # ax.plot(rsArr[j,:-1,s]*1e6, Es[j,:-1,s]-ref, color = colsBase[j], linestyle='dashed', zorder=20, linewidth=1)

    # Set the rest of the plot stuff
    ax.set_xlabel("Distance ($\\mu m$)")
    ax.set_ylabel("Eigenenergy (MHz)")
    #ax.set_title('Rabi Frequency 6 MHz')
    ax.set_ylim(ymin,ymax)
    plt.show()




def GenerateSweepPlots(Es, rsArr, eigenstateInds, fosArr,stateIndices):
    ### This cell does the plotting.
    # Parameters
    # Determine the colors for the dots for our plot in advance
    # colsBase is the "base" (full opacity) color specified in RGB from 0 to 1
    colsBase = np.array([[0,0,0],[0,0,1],[0,0.4,0]])

    stateLenVec = np.zeros(len(dfs))
    for i in range(len(stateIndices)):
        stateLenVec[i] = len(stateIndices[i])

    Eb = 0.75
    # Refernece flag: this Boolean sets the offsets with which the
    # eigenvalues are plotted. If 0, all the states are plotted with the
    # same offset, equal to the energy of the SOI the dataset was
    # created with. If 1, every eigenstate of every dataset is
    # individually offset so that its energy is zero at large distance.
    refBool = 0

    # Make the plot
    # Create the array of colors for the dots
    # Indices: {data file, state, r, RGBA}
    cols = np.zeros((len(dfs), int(max(stateLenVec)), len(rs)-1, 4))
    for j in range(len(dfs)):
        for s in range(len(stateIndices[j])):
            cols[j,s,:,0] = colsBase[j][0]
            cols[j,s,:,1] = colsBase[j][1]
            cols[j,s,:,2] = colsBase[j][2]
            # This is opacity; if you want no opacity changes just set this to 1
            cols[j,s,:,3] = fosArr[j,:,s][:-1]

    # # This is a filled in array of r values for plotting
    # # fit lines, shaded boundaries, etc.
    # contRs = np.linspace(rs[0],rs[-2],int((rs[-2]-rs[0])*1e6*2+1))

    # Make the plot
    fig, ax = plt.subplots(figsize=(12,8))
    for j in range(len(dfs)):
        for s in range(stateNums[j]):
            # Set the reference point
            ref = refBool*Es[j,-1,s]
            # Plot the eigenvalues
            ax.scatter(rs[:-1]*1e6, Es[j,:-1,s]-ref, color=cols[j,s], marker='o', zorder=25, linestyle='-', linewidth=1)
            # Plot the blockade bands if desired
            ax.fill_between(contRs*1e6, [Es[j,-1,s]-ref-Eb]*len(contRs), [Es[j,-1,s]-ref+Eb]*len(contRs), color=colsBase[j], alpha=(1-refBool)*0.1 + (refBool)*0.05)
            # ax.plot(contRs*1e6, [Es[j,-1,s]-ref]*len(contRs), linestyle='--', color=colsBase[j], alpha=(1-refBool)*0.2 + (refBool)*0.1)
            # Plot the fit lines if desired
            # ax.plot(contRs*1e6, fitParams[j,s,0]/contRs**3 + fitParams[j,s,1]/contRs**6 + fitParams[j,s,2] - Es[j,-1,0],
            #        color = colsBase[j], linestyle='dashed', zorder=20)
            # Plot lines between eigenvalues if desired
            #ax.plot(rs[:-1]*1e6, Es[j,:-1,s]-ref, color = colsBase[j], linestyle='dashed', zorder=20, linewidth=1)

    # Build the legend using artists independent of any of the points
    # for proper opacity appearance
    # legendArtists = [0]*len(colsBase)
    # for j in range(len(dfs)):
    #     legendArtists[j] = mlines.Line2D([], [], color=colsBase[j], marker='o', markersize=8, linewidth=0, label=legendLbls[j])
    # ax.legend(handles=legendArtists, fontsize=16)

    # Set the rest of the plot stuff
    ax.set_xlabel("Distance ($\\mu m$)")
    ax.set_ylabel("Eigenenergy (MHz)")
    #ax.set_title('Rabi Frequency 6 MHz')
    ax.set_ylim(-8,8)
    plt.show()