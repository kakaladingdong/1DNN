import numpy as np
import scipy.io as sio
import os

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def average(args,num):
    result=args.result
    oa,aa,kappa=0,0,0
    for i in range(num):
        data=sio.loadmat(os.path.join(result,str(i),'result.mat'))
        oa+=data['oa'][0][0]
        aa+=data['aa'][0][0]
        kappa+=data['kappa'][0][0]
    oa_ave=oa/num
    aa_ave=aa/num
    kappa_ave=kappa/num

    print('oa_ave:',oa_ave,'aa_ave:',aa_ave,'kappa_ave:',kappa_ave)

    result=result+'/ave'
    if not os.path.exists(result):
        os.mkdir(result)
    sio.savemat(os.path.join(result,'ave.mat'),{'oa_ave':oa_ave,'aa_ave':aa_ave,'kappa_ave':kappa_ave})

