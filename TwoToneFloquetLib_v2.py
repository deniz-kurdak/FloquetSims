###########################################################################
#
# Two-Atom Rydberg Two Tone Floquet Library
# Deniz Kurdak
# May 2025
# 
# The single floquet methods of this library were written by Patrick Banner
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


###########################################################################
#
# Atom Classes
#
###########################################################################

class oneAtomState:
    n=0; l=0; j=0; mj=0;
    q1=0;     # Fourier component of field 1 for Floquet state
    q2=0;     # Fourier component of field 2 for Floquet state


    # Initialization with n, l, j, mj, and optional q_in for the Fourier component
    def __init__(self,n_in,l_in,j_in,mj_in,q1_in = 0, q2_in = 0):
        self.n = int(n_in); self.l = int(l_in); self.j = j_in; self.mj = mj_in
        self.q1 = int(q1_in); self.q2 = int(q2_in)
    
    # Convert to printable string
    def __str__(self):
        return "({:.0f}, {:.0f}, {:.1f}, {:.1f}, q1={:.0f}, q2={:.0f})".format(self.n, self.l, self.j, self.mj, self.q1, self.q2)
    
    # Convert to Python list (mostly for saving to files)
    def tolist(self):
        return [self.n, self.l, self.j, self.mj, self.q1, self.q2]
    
    # Checking equality between two states
    def __eq__(self, other):
        if self.n == other.n and self.l == other.l and self.j == other.j and self.mj == other.mj and self.q1 == other.q1 and self.q2 == other.q2:
            return True
        else:
            return False

class twoAtomState:
    n1=0; l1=0; j1=0; mj1=0;
    n2=0; l2=0; j2=0; mj2=0;
    q1=0;     # Fourier component of field 1 of atom 1 
    q2=0     # Fourier component of field 2 of atom 1 


    # Initialization with n, l, j, mj, and optional q_in for the Fourier component
    def __init__(self,n1_in,l1_in,j1_in,mj1_in,n2_in,l2_in,j2_in,mj2_in,q1_in = 0,q2_in = 0):
        self.n1 = int(n1_in); self.l1 = int(l1_in); self.j1 = j1_in; self.mj1 = mj1_in
        self.n2 = int(n2_in); self.l2 = int(l2_in); self.j2 = j2_in; self.mj2 = mj2_in
        self.q1 = int(q1_in); self.q2 = int(q2_in)

    # Call as twoAtomState.fromOneAtomStates(s1, s2) where s1, s2 are oneAtomStates
    # Returns the relevant two-atom states
    @classmethod
    def fromOneAtomStates(cls, s1, s2):
        if (s1.q1 != s2.q1) or (s1.q2 != s2.q2):
            raise Exception("The q values of the states must match.")
        return cls(s1.n, s1.l, s1.j, s1.mj, s2.n, s2.l, s2.j, s2.mj, s1.q1, s1.q2)
    
    # Convert to printable string
    def __str__(self):
        return "(({:.0f}, {:.0f}, {:.1f}, {:.1f}), ({:.0f}, {:.0f}, {:.1f}, {:.1f}), q1={:.0f}, q2={:.0f})".format(self.n1, self.l1, self.j1, self.mj1, self.n2, self.l2, self.j2, self.mj2, self.q1, self.q2)
    
    # Convert to Python list (mostly for saving to files)
    def tolist(self):
        return [self.n1, self.l1, self.j1, self.mj1, self.n2, self.l2, self.j2, self.mj2, self.q1, self.q2]
    
    # Checking equality between two states
    def __eq__(self, other):
        if self.n1 == other.n1 and self.l1 == other.l1 and self.j1 == other.j1 and self.mj1 == other.mj1 and self.n2 == other.n2 and self.l2 == other.l2 and self.j2 == other.j2 and self.mj2 == other.mj2 and self.q1 == other.q1 and self.q2 == other.q2:
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

def getOneAtomFloquetStateList(stateList, q1Max, q2Max):
    '''
    Given a list of one-atom states (e.g. from getStateList()), this returns
    a list of oneAtomStates of all the one-atom Floquet states (i.e. with
    correctly-assigned q values). 
    Inputs: stateList, the Python list of oneAtomState objects, and qMax,
            specifying the range of Fourier components to include (the range
            is such that all states with -qMax <= q <= qMax will be included).
    '''
    
    if q1Max < 0 or q2Max < 0:
        print("You need to specify qMax >= 0.")
        return None
    
    # Let's make the Floquet state list
    stateNum = len(stateList)
    floquetStateList = []
    for q1 in getRange(-q1Max,q1Max):
        for q2 in getRange(-q2Max, q2Max):
            for j in range(stateNum):
                floquetStateList.append(oneAtomState(stateList[j].n,stateList[j].l,stateList[j].j,stateList[j].mj,q1_in=q1,q2_in=q2))
            
    return floquetStateList

# Given an atomic state list, make the two-atom Floquet states, whose Fourier indices go from -qMax to qMax inclusively
def getTwoAtomFloquetStateList(oneAtomStateList, q1Max, q2Max):
    '''
    Given a list of one-atom states (e.g. from getStateList()), this returns
    a list of twoAtomStates of all the two-atom Floquet states (i.e. with
    correctly-assigned q values, and every choice of two of the one-atom states
    in the input list). 
    Inputs: oneAtomStateList, the Python list of oneAtomState objects, and qMax,
            specifying the range of Fourier components to include (the range
            is such that all states with -qMax <= q <= qMax will be included).
    '''
    if q1Max < 0 or q2Max < 0:
        print("You need to specify qMax >= 0.")
        return None
    
	# Let's make the Floquet state list
    stateNum = len(oneAtomStateList)
    floquetStateList = []
    for q1 in getRange(-q1Max, q1Max):
        for q2 in getRange(-q2Max, q2Max):
            for j in range(stateNum):
            	for k in range(stateNum):
                    floquetStateList.append(twoAtomState(
                        oneAtomStateList[j].n,oneAtomStateList[j].l,oneAtomStateList[j].j,oneAtomStateList[j].mj,
                        oneAtomStateList[k].n,oneAtomStateList[k].l,oneAtomStateList[k].j,oneAtomStateList[k].mj,
                        q1_in=q1, q2_in=q2))
            
    return floquetStateList


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

def getH_diag_twoAtom_fromBase(H_diag_twoAtom_base, q1Max, q2Max, rf_freq1, rf_freq2):
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
    H = sp.lil_matrix((H_diag_twoAtom_base.shape[0]*(2*q1Max + 1)*(2*q2Max + 1), H_diag_twoAtom_base.shape[0]*(2*q1Max + 1)*(2*q2Max + 1)))
    diagElements0 = np.array([H_diag_twoAtom_base.diagonal(0)])
    diagElements = np.array([diagElements0 + q1*rf_freq1 + q2*rf_freq2 for q2 in np.arange(-q2Max, q2Max+1, 1) for q1 in np.arange(-q1Max, q1Max+1, 1)]).flatten()
    H.setdiag(diagElements)
    return H

def getH_diag_twoAtom(oneAtomStateList, soi, B, qMax, rf_freq1, rf_freq2):
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
    return getH_diag_twoAtom_fromBase(H2, q1Max, q2Max, rf_freq1, rf_freq2)


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

def getH_E_DC_twoAtom_fromBase(H_E_DC_twoAtom_base, q1Max, q2Max):
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
    H = sp.lil_matrix((dim*(2*q1Max + 1)*(2*q2Max + 1), dim*(2*q1Max + 1)*(2*q2Max + 1)), dtype='complex_')
    # Copy the base H into the diagonal blocks
    for n in np.arange(0, (2*q1Max+1)*(2*q2Max+1)):
        H[n*dim:(n+1)*dim, n*dim:(n+1)*dim] = H_E_DC_twoAtom_base
    return H

def getH_E_DC_twoAtom(oneAtomStateList, soi, pol, q1Max, q2Max, E = 1):
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
    return getH_E_DC_twoAtom_fromBase(H2, q1Max, q1Max)


###########################################################################
#
# Getting Hamiltonians - AC E-field
#
###########################################################################

def getH_E_AC_oneAtom_base_field1(oneAtomStateList, soi, pol1, E1):
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
    polN1 = pol1/np.linalg.norm(pol1)
    
    for x in range(stateNum):
        for y in range(x+1, stateNum):
            s1 = oneAtomStateList[x]; s2 = oneAtomStateList[y];

            # # Flip if needed
            # if (atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j) < 0):
            #     sTemp = s2; s2 = s1; s1 = sTemp

            # Only apply if selection rules are satisfied
            if (s1 != s2) and (abs(s1.j - s2.j) <= 1) and (abs(s1.mj - s2.mj) <= 1) and (abs(s1.l - s2.l) == 1):
                q = int((s2.mj - s1.mj))
                eff_E1 = E1 * polN1[1+q]
                #H[x,y] = atom.getRabiFrequency2(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,q,eff_E)/(2*pi)/2
                H[x,y] = atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,s2.mj - s1.mj)*(C_e*a0/C_h)*eff_E1/2 
                H[y,x] = np.conjugate(H[x,y])
            
    return stateNum, H

def getH_E_AC_twoAtom_base_field1(H_E_AC_oneAtom_field1):
    '''
    [Primarily a utility method]
    This method forms the AC E-field portion of the two-atom Hamiltonian from the
    AC E-field portion of the one-atom Hamiltonian by simply Kronecker summing. The
    returned matrix is a scipy.sparse.lil_matrix object.
    '''
    return sp.kronsum(H_E_AC_oneAtom_field1, H_E_AC_oneAtom_field1, format="lil")

def getH_E_AC_twoAtom_fromBase_field1(H_E_AC_oneAtom_base_field1, q1Max):
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
    dim = H_E_AC_oneAtom_base_field1.shape[0]
    H = sp.lil_matrix((dim*(2*q1Max + 1), dim*(2*q1Max + 1)), dtype='complex_')
    for n in np.arange(0, (2*q1Max+1)-1):
        H[n*dim:(n+1)*dim, (n+1)*dim:(n+2)*dim] = H_E_AC_oneAtom_base_field1
    H2 = H.copy() + H.copy().conjugate().transpose()
    return H2

def getH_E_AC_twoAtom_fromBase_field2(H_E_AC_oneAtom_base_field2, q2Max):
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
    dim = H_E_AC_oneAtom_base_field2.shape[0]
    H = sp.lil_matrix((dim*(2*q2Max + 1), dim*(2*q2Max + 1)), dtype='complex_')
    for n in np.arange(0, (2*q2Max+1)):
        H[n*dim:(n+1)*dim, (n)*dim:(n+1)*dim] = H_E_AC_oneAtom_base_field2
    H2 = H.copy() + H.copy().conjugate().transpose()
    return H2

def getH_E_AC_twoAtom_field1(oneAtomStateList, soi, pol1, q1Max, E1):
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
    _, H1 = getH_E_AC_oneAtom_base_field1(oneAtomStateList, soi, pol1, E1)
    H2 = getH_E_AC_twoAtom_base_field1(H1)
    return getH_E_AC_twoAtom_fromBase_field1(H2, q1Max)

def getH_E_AC_twoAtom_q12_field1(oneAtomStateList, soi, pol1, q1Max, q2Max, E1):
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
    _, H1 = getH_E_AC_oneAtom_base_field1(oneAtomStateList, soi, pol1, E1)
    H2 = getH_E_AC_twoAtom_base_field1(H1)
    H2field1_q1 =  getH_E_AC_twoAtom_fromBase_field1(H2, q1Max)

    dim = H2field1_q1.shape[0]
    H = sp.lil_matrix((dim*(2*q2Max + 1), dim*(2*q2Max + 1)), dtype='complex_')
    for n in np.arange(0, (2*q2Max+1)):
        H[n*dim:(n+1)*dim, (n)*dim:(n+1)*dim] = H2field1_q1
    H2_field1_q12 = H.copy() + H.copy().conjugate().transpose()
    return H2_field1_q12

def getH_E_AC_twoAtom_q12_field2(oneAtomStateList, soi, pol2, q1Max, q2Max, E2):
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
    _, H1 = getH_E_AC_oneAtom_base_field1(oneAtomStateList, soi, pol2, E2)
    H2 = getH_E_AC_twoAtom_base_field1(H1)
    H2field2_q2 =  getH_E_AC_twoAtom_fromBase_field2(H2, q2Max)

    dim = H2field2_q2.shape[0]
    H = sp.lil_matrix((dim*(2*q1Max + 1), dim*(2*q1Max + 1)), dtype='complex_')
    for n in np.arange(0, (2*q1Max+1)-1):
        H[n*dim:(n+1)*dim, (n+1)*dim:(n+2)*dim] = H2field2_q2
    H2_field2_q12 = H.copy() + H.copy().conjugate().transpose()
    return H2_field2_q12




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
                sn = np.sign(atom.getTransitionFrequency(s1.n,s1.l,s1.j,s2.n,s2.l,s2.j))
                MDref[x,y] = atom.getDipoleMatrixElement(s1.n,s1.l,s1.j,s1.mj,s2.n,s2.l,s2.j,s2.mj,s2.mj - s1.mj)*(C_e*a0)
                MDref[y,x] = np.conjugate(MDref[x,y])#*(sn)**(s2.mj-s1.mj)

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

    # This creates a list of two-atom states, with the q1 and q2 indices = 0
    twoAtomSL = getTwoAtomFloquetStateList(oneAtomStateList, 0, 0)
    # This creates the dipole-dipole Hamiltonian in the two atom basis
    # Hence the stateNum square number of elements
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

def getH_dd_twoAtom_fromBase(H_dd_twoAtom_base, q1Max, q2Max):
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
    H = sp.lil_matrix((dim*(2*q1Max + 1)*(2*q2Max + 1), dim*(2*q1Max + 1)*(2*q2Max + 1)), dtype='complex_')
    # Copy the input into (2*q1Max+1)*(2*q2Max+1) blocks on the diagonal
    for n in np.arange(0, (2*q1Max+1)*(2*q2Max+1)):
        H[n*dim:(n+1)*dim, n*dim:(n+1)*dim] = H_dd_twoAtom_base
    return H


###########################################################################
#
# Utility Functions
#
###########################################################################

# When using this, be sure to specify the Fourier component of the Floquet state!!!
# Several operating modes depending on SAS:
#     0 - find exactly and only soi
#     ±1 - find soi ± the soi with the atom states flipped
#     +2 - take the one-atom states |a> and |b> from soi and find
#          (v[0] |a> + dot[1] |b>) × (dot[0] |a> + v[1] |b>)
#     +3 - v should be a list representing the entire two-atom basis
#          and this function will find the overlap with the state
#          represented by the vector dot in that basis
def getFracsOfState(oneAtomStateList, q1Max, q2Max, evecs, soi, SAS, stateInds = [], v = [], psfBool=0):
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

    # Get the two-atom Floquet state list
    twoAtomFSL = np.array(getTwoAtomFloquetStateList(oneAtomStateList, q1Max , q2Max), dtype=object)
    if (len(stateInds) != 0):
        twoAtomFSL = np.array(twoAtomFSL.copy(), dtype=object)[stateInds]

    # We will use v to take a dot product with our state, so based on SAS, set v appropriately
    if (SAS == 0):
        soiInd = np.where(twoAtomFSL == soi)[0][0]
        v = np.zeros(len(twoAtomFSL)); v[soiInd] = 1
    elif (abs(SAS) == 1):
        soiInd = np.where(twoAtomFSL == soi)[0][0]
        soi2 = twoAtomState(soi.n2,soi.l2,soi.j2,soi.mj2,soi.n1,soi.l1,soi.j1,soi.mj1,soi.q)
        soiInd2 = np.where(twoAtomFSL == soi2)[0][0]
        v = np.zeros(len(twoAtomFSL))
        v[soiInd] = 1/np.sqrt(2); v[soiInd2] = SAS*1/np.sqrt(2)
    elif (SAS == 2):
        s1 = twoAtomState(soi.n1,soi.l1,soi.j1,soi.mj1,soi.n1,soi.l1,soi.j1,soi.mj1,soi.q+psfBool)
        s2 = twoAtomState(soi.n1,soi.l1,soi.j1,soi.mj1,soi.n2,soi.l2,soi.j2,soi.mj2,soi.q)
        s3 = twoAtomState(soi.n2,soi.l2,soi.j2,soi.mj2,soi.n1,soi.l1,soi.j1,soi.mj1,soi.q)
        s4 = twoAtomState(soi.n2,soi.l2,soi.j2,soi.mj2,soi.n2,soi.l2,soi.j2,soi.mj2,soi.q-psfBool)
        si1 = np.where(twoAtomFSL == s1)[0][0]; si2 = np.where(twoAtomFSL == s2)[0][0];
        si3 = np.where(twoAtomFSL == s3)[0][0]; si4 = np.where(twoAtomFSL == s4)[0][0]
        a = v[0]/np.linalg.norm(v); b = v[1]/np.linalg.norm(v)
        v = np.zeros(len(twoAtomFSL))
        v[si1] = np.conj(a)*a; v[si2] = np.conj(b)*a; v[si3] = np.conj(a)*b; v[si4] = np.conj(b)*b
    elif (SAS == 3):
        # v is already what we need for the next part
        v = v/np.linalg.norm(v)
        pass

    # Now take the dot product
    cs = np.abs(np.dot(v, np.transpose(evecs)))**2
    # I think this can made into one line because NumPy is smart, but it's fine
    #for i, vec in enumerate(evecs):
    #    cs[i] = np.abs(np.dot(v, vec))**2
    return cs

def getStateOverlaps(states, qMax, a, b, evals, evecs, stateInds=[], thres=0.01, verbose=True):
    '''
    This function PRINTS state overlap data in human-readable format. 
    Specifically, it selects an eigenvector and shows the coefficients for that
    eigenstate in the bare two-atom Floquet basis.
    Inputs:
        states: a Python list of oneAtomState objects that were the one-atom
                basis used when generating the input eigenvectors (so this
                function can reconstruct the basis that those eigenvectors are
                expressed in)
        qMax: the maximum Fourier component used when generating the input eigenvectors
              (so this function can reconstruct the basis that those eigenvectors are expressed in)
        a: the key to get the eigenvectors from the dictionary evecs
        b: the eigenvector to look at
        evals: the DICTIONARY of eigenvalues of the system, as generated by most of the other
               functions in this library (only used for printing additional info)
        evecs: the DICTIONARY of eigenvectors of the system, as generated by most of the other
               functions in this library, with keys 0, 1, 2, etc (integers, not strings)
               corresponding to rf frequencies or other parameters
        stateInds (optional): a list of indices of states to keep in the two-atom Floquet basis; if
                   keeping all of the states, use the empty set [] (default [])
        thres (optional): a threshold |coeff|^2 above which to print info and below which to
                   not (default 0.01)
    Outputs: None (only printing)
    '''
    stateListF1 = np.array(getTwoAtomFloquetStateList(states, qMax), dtype=object)
    if (len(stateInds) != 0):
        stateListF1 = stateListF1[stateInds]
    
    inds = np.where(np.abs(evecs[a][b])**2 > thres)[0]

    stateInds = np.zeros(len(inds))
    evecOverlaps = np.zeros(len(inds))
    states = np.array([], dtype='object')
    #if verbose:
    print("Eigenenergy:", np.real(evals[a][b])/1e6, "MHz")
    for i, ind in enumerate(inds):
        stateInds[i] = ind
        evecOverlaps[i] = evecs[a][b][ind]
        states = np.append(states, stateListF1[ind])
        #if verbose:
        print(("{:.0f}\t{:.4f}\t{:.4f}\t" + str(np.asarray(stateListF1, dtype='object')[ind])).format(ind, evecs[a][b][ind], np.abs(evecs[a][b][ind])**2))

    #return evals[a][b], stateInds, evecOverlaps, states
        



###########################################################################
#
# Functions for sweeping the AC frequency
#
###########################################################################

def sweepACfreq(low_n, high_n, max_l, q1Max, q2Max, soi, b_field, e_field, rf_sweep_freqs, rf_fixed_freq, rf_sweep_inten, rf_fixed_inten, dc_polarization, ac_polarization1, ac_polarization2, theta, phi, DDmult, stateInds=[], keepEVNum = 0, saveFile = "", progress_notif = 10, stateList=0, verbose=True):
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
        theta, phi: the polar and azimuthal angles of the interatomic axis between the atoms,
                    referenced to the quantization axis
        DDmult: the multiplier in front of the dipole-dipole matrix,
                which should be either zero or 1/r^3 where r is the distance between the atoms, in m
                (default 1).
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
    dimH = len(oneAtomBareStates)**2 * (2*q1Max + 1) * (2*q2Max + 1)
    if verbose:
        print("Made the state lists. Total of {:.0f} states.".format(dimH), end=" ")

    # Get the DC Hamiltonian.
    H_DC = sp.lil_matrix((dimH, dimH))
    # If e_field = 0, no need to compute
    if (e_field != 0):
        int_H_DC = getH_E_DC_twoAtom(oneAtomBareStates, soi, dc_polarization, q1Max, q2Max, E = e_field)
        if verbose:
            print("Made DC Hamiltonian.", end=" ")
    # Get the AC Hamiltonians.
    H_AC_1 = getH_E_AC_twoAtom_q12_field1(oneAtomBareStates, soi, ac_polarization1, q1Max, q2Max, np.sqrt(2*rf_sweep_inten/(C_c*epsilon0)))
    H_AC_2 = getH_E_AC_twoAtom_q12_field2(oneAtomBareStates, soi, ac_polarization2, q1Max, q2Max, np.sqrt(2*rf_fixed_inten/(C_c*epsilon0)))
    if verbose:
        print("Made AC Hamiltonian.")

    # Get the *base* diagonal matrix... this has to be modified for every RF frequency,
    # but we can do a lot of the base computation in advance
    _, H_diag_oneAtom_base = getH_diag_oneAtom_base(oneAtomBareStates, soi, b_field)
    H_diag_twoAtom_base = getH_diag_twoAtom_base(H_diag_oneAtom_base)

    # Finally get Hdd
    _, Hdd_base = getH_V_dd_twoAtom_base(oneAtomBareStates, theta, phi, MDref = [])
    Hdd = DDmult*getH_dd_twoAtom_fromBase(Hdd_base, q1Max, q2Max)
    Hddtemp = np.abs(Hdd.toarray())/1e6
    if verbose:
        print("Made Hdd Hamiltonian.", end=" ")
        print("Average Hdd nonzero element is {:.4f} MHz;".format(np.mean(Hddtemp, where=(Hddtemp != 0))), end=" ")
        print("Max Hdd element is {:.4f} MHz".format(np.max(Hddtemp)))
    del Hddtemp

    eigenvalues = {}
    eigenvectors = {}
    if verbose:
        print("Starting analysis of the Hamiltonians...")
    # Start the sweep.
    for r, rf_sweep_freq in enumerate(rf_sweep_freqs):
        # Get the diagonals for the RF frequency
        H_diag = getH_diag_twoAtom_fromBase(H_diag_twoAtom_base, q1Max, q2Max, rf_sweep_freq, rf_fixed_freq)
        # Total Hamiltonian
        Htot = (H_diag + H_DC + H_AC_1 + H_AC_2 + Hdd).toarray()
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
            dset.attrs["n0"] = low_n; dset.attrs["n1"] = high_n; dset.attrs["max_l"] = max_l; dset.attrs["max_q1"] = q1Max; dset.attrs["max_q2"] = q2Max;
            dset.attrs["soi"] = soi.tolist(); dset.attrs["si"] = stateInds;
            dset.attrs["B"] = b_field;
            dset.attrs["E"] = e_field; dset.attrs["E_DC_pol"] = dc_polarization;
            dset.attrs["rf_sweep_freq"] = rf_sweep_freq; dset.attrs["rf_fixed_freq"] = rf_fixed_freq;
            dset.attrs["rf_sweep_inten"] = rf_sweep_inten; dset.attrs["rf_fixed_inten"] = rf_fixed_inten;
            dset.attrs["E_AC_pol1"] = ac_polarization1; dset.attrs["E_AC_pol2"] = ac_polarization2;
            dset.attrs["theta"] = theta; dset.attrs["phi"] = phi; dset.attrs["DDmult"] = DDmult;
            dset.attrs["evals"] = np.real(eigenvalues[r]);

        # Update on progress
        if (verbose and (progress_notif == -1) or ((r+1)%progress_notif == 0)):
            print("{:.0f} done...".format(r+1), end=" ")

    # Make sure to close the file properly
    if (saveFile != ""):
        f.close()
    
    return oneAtomBareStates, eigenvalues, eigenvectors, Htot, H_DC, H_AC_1 + H_AC_2, Hdd_base

def sweepACfreq_plot(states, SASs, q1Max, q2Max, rf_sweep_freqs, plotLbls, cmaps, stateList, stateInds, evals, evecs, ylim=[], title="", s0 = 10, locArg='best', plotAll = True, savefig=""):
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
        plotCircs.append(mlines.Line2D([], [], color=cmaps[j](256), marker='.', linestyle='None', markersize=8, label=plotLbls[j]))

    # Do the state overlap calculation and plotting
    fig, ax = plt.subplots(figsize=(15,6))
    for i in range(len(rf_sweep_freqs)):
        evNum = len(evals[i])
        if plotAll:
            sc0 = ax.scatter(np.array([rf_sweep_freqs[i]]*evNum)/1e6, evals[i]/1e6, c=[[0,0,0,0.1]], s=s0)
        for j in range(len(plotLbls)):
            colFracs = getFracsOfState(stateList, q1Max, q2Max, evecs[i], states[j], SASs[j], stateInds=stateInds)
            sc = ax.scatter(np.array([rf_sweep_freqs[i]]*evNum)/1e6, evals[i]/1e6, c=colFracs, cmap=cmaps[j], s=s0, vmin=0, vmax=1)

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














