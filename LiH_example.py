#%%
import numpy as np
import matplotlib.pyplot as plt
from thewalrus import perm
from scipy.optimize import minimize
from scipy.linalg import expm
import joblib as jb
from qiskit.quantum_info import SparsePauliOp
from numba import njit,prange
#%%
def linear_optics(m,n,para):
    dype = np.complex64
    occupied=np.append(np.arange(int(n/2)),np.arange(int(m/2),int(m/2)+int(n/2)))
    virtual=np.append(np.arange(int(n/2),int(m/2)),np.arange(int(m/2)+int(n/2),m))
    H=np.zeros((m,m),dtype=dype)
    for i in range(m):
        H[i,i]=para[i]
    count=0
    for i in occupied:
        for j in virtual:
            H[i,j]=para[m+count]+1j*para[m+count+1]
            H[j,i]=para[m+count]-1j*para[m+count+1]
            count+=2
    num=0
    for i in range(m-n):
        for j in range(i):
            H[virtual[i],virtual[j]]=para[m+count+num]+1j*para[m+count+num+1]
            H[virtual[j],virtual[i]]=para[m+count+num]-1j*para[m+count+num+1]
            num+=2
    result=expm(-1j*H)
    return result

def lopanum(m,n):
    result=m
    occupied=np.append(np.arange(int(n/2)),np.arange(int(m/2),int(m/2)+int(n/2)))
    virtual=np.append(np.arange(int(n/2),int(m/2)),np.arange(int(m/2)+int(n/2),m))
    for i in occupied:
        for j in virtual:
            result+=2
    for i in range(m-n):
        for j in range(i):
            result+=2
    return result

def linear_fermion(m,para):
    dype = np.complex64
    H=np.zeros((m,m),dtype=dype)
    for i in range(m):
        H[i,i]=para[i]
    count=0
    for i in range(m):
        for j in range(i):
            H[i,j]=para[m+count]+1j*para[m+count+1]
            H[j,i]=para[m+count]-1j*para[m+count+1]
            count+=2
    result=expm(-1j*H)
    return result

def decimto(num, n1, n2):
    mlist = np.zeros(n1, dtype=int)
    i = 0
    while True:
        mlist[i] = num % n2
        num = num // n2
        i += 1
        if num == 0:
            break
    result = np.zeros(n1, dtype=int)
    result[:] = mlist[::-1]
    return result

def BS(idin,idout,U):
    idi=list(np.where(np.array(idin)==1)[0])
    ido=list(np.where(np.array(idout)==1)[0])
    result=perm(U[:,idi][ido], method='ryser')
    return result

def FS(idin,idout,U):
    idi=list(np.where(np.array(idin)==1)[0])
    ido=list(np.where(np.array(idout)==1)[0])
    result=np.linalg.det(U[:,idi][ido])
    return result

def BSlist(hf_state,basis_2_posi,Ub):
    hf=np.where(np.array(hf_state)==1)[0]
    result=np.array([perm(Ub[:,hf][i], method='ryser') for i in basis_2_posi])
    return result

@njit
def FSlist(basis_2_posi,Uf):
    dype = np.complex64
    result=np.zeros((basis_2_posi.shape[0],basis_2_posi.shape[0]),dtype=dype)
    for i in range(basis_2_posi.shape[0]):
         for hi in range(basis_2_posi.shape[0]):
            result[i,hi]=np.linalg.det(Uf[:,basis_2_posi[i]][basis_2_posi[hi]])
    return result

@njit(parallel=True, fastmath=True)
def expectation(H_value,H_id,basis_2_posi_length,bslist,fslist): 
    result=0
    for i in prange(basis_2_posi_length):
        for j in prange(basis_2_posi_length):
            for h in prange(H_id.shape[0]):
                result+=np.real(np.conj(bslist[i])*bslist[j]*fslist[j,H_id[h,1]]*np.conj(fslist[i,H_id[h,0]])*H_value[h])
    return result

def eigen(x):
    w=np.linalg.eigh(x)[0]
    result = w
    return result

def openfermion_to_qulacs(file):
    file_list=list(open(file))
    paulilist=[x.split(' ',1)[1].strip(' \n').replace('0','I').replace('1','X').replace('2','Y').replace('3','Z').replace(' ', '') for x in file_list]
    coefflist=[complex(x.split(' ',1)[0]) for x in file_list]
    hm=SparsePauliOp(paulilist,coeffs=coefflist).to_matrix()
    energy=eigen(hm)[0]
    length=len(paulilist)
    return (hm,energy,length)

def legalbasis(n,m):
    basis_10=[]
    for i in range(2**m):
        if np.sum(decimto(i,m,2))==n:
            basis_10.append(i)
    basis_10=np.array(basis_10)
    basis_2=np.array([decimto(i,m,2) for i in basis_10])
    basis_2_posi=np.array([np.where(np.array(basis_2[i])==1)[0] for i in range(basis_10.size)])
    basis_tensor=np.array([[i,j] for i in basis_10 for j in basis_10])
    return (basis_10,basis_2,basis_2_posi,basis_tensor)

def H_dict_generate(hm,basis_tensor,basis_10):
    va=[hm[it[0],it[1]] for it in basis_tensor]
    va=np.array(va)
    newva=va[np.where(va!=0)[0]]
    newbasis=basis_tensor[np.where(va!=0)[0]]
    newid=np.array([[np.where(basis_10==h[0])[0][0],np.where(basis_10==h[1])[0][0]] for h in newbasis])
    result=(newid,newbasis,newva)
    return result

def projector(hf_state,basis_2,Ub):
    res=[np.abs(BS(hf_state,it,Ub))**2 for it in basis_2]
    result=np.sum(res)
    return result

#%% LiH bond length=1.0 example, using BS&HF
if __name__ == '__main__':
    dype = np.complex64
    cpu_num=1
    penality=0.0005 #penality term
    #basic
    n=2 # number of electrons
    m=6 # number of orbitalss
    onum=lopanum(m,n)
    fnum=m+2*np.sum(np.arange(1,m))
    pnum=onum+fnum
    hf_state=np.array([1,0,0,1,0,0]) # initial state
    #setting
    repetition=1
    bond='1.0' #bond length
    ge=-7.782242403 #ground energy
    paralist=0.2*np.random.rand(repetition,pnum)
    ####
    (basis_10,basis_2,basis_2_posi,basis_tensor)=legalbasis(n,m)
    #Hamiltonian file import, generated from openfermion
    (hm,energy,paulilength)=openfermion_to_qulacs('LiH_6_R'+bond+'.txt') 
    (h_id,h_basis,h_value)=H_dict_generate(hm,basis_tensor,basis_10)  
    H_basis=np.array_split(h_basis, cpu_num)
    H_id=np.array_split(h_id, cpu_num)
    H_value=np.array_split(h_value, cpu_num)
    basis_2_posi_length=basis_2_posi.shape[0]
    print("Bond distance: ", bond)
    print("Number of qubits/spin-orbitals: ",m)
    print("Number of electrons: ",n)
    print("Number of Pauli terms: ",paulilength)
    print("Matrix Ground state energy: ",energy) 
    print("Ground state energy: ",ge)  
    print("Number of parameters: ",pnum)
    for i in range(repetition):
        expt_history=[]
        proj_history=[]
        def expt(parain):
            Ub=linear_optics(m,n,parain[0:onum])
            Uf=linear_fermion(m,parain[onum:pnum])
            bslist=BSlist(hf_state,basis_2_posi,Ub)
            fslist=FSlist(basis_2_posi,Uf)
            project=np.sum(np.abs(bslist)**2) 
            proj_history.append(project) 
            def f(i):
                return expectation(H_value[i],H_id[i],basis_2_posi_length,bslist,fslist)
            exp_sect=jb.Parallel(n_jobs=cpu_num,backend='loky')(jb.delayed(f)(i) for i in range(cpu_num))  
            expt_tot=np.sum(exp_sect)/project
            expt_history.append(expt_tot)
            if len(expt_history)%2000==0:
                print("Distance: ",expt_history[-1]-ge)
            result=expt_tot-penality*project
            return result
        print("Experiment: ", i)
        parain=paralist[i]
        opt_result = minimize(expt, parain, method='L-BFGS-B', tol=1e-15,options={'maxfun': 20000, 'maxiter':20000, 'ftol':2.220446049250313e-15, 'eps':1e-04,})
        plt.plot(expt_history)
        plt.hlines(ge, xmin=0, xmax=len(expt_history), label='ground', color='black', linestyle='dashed')
        plt.xlabel("Iteration")
        plt.ylabel("Energy expectation value")
        plt.legend()
        plt.show()
        chem = np.log10(np.array(expt_history) - ge)
        chem_acc = np.log10(1.6 * 10 ** (-3))
        plt.plot(chem)
        plt.hlines(chem_acc, xmin=0, xmax=len(expt_history), label='chemical accuracy', color='black',
                    linestyle='dashed')
        plt.xlabel("Iteration")
        plt.ylabel("Energy difference")
        plt.legend()
        plt.show()
        plt.plot(proj_history)
        plt.hlines(1, xmin=0, xmax=len(proj_history), label='1', color='black', linestyle='dashed')
        plt.xlabel("Iteration")
        plt.ylabel("Projection Ratio")
        plt.legend()
        plt.show()
# %%
