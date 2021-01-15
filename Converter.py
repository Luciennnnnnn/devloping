import scipy.io as io
import numpy as np

mat = io.loadmat('TPRS.mat')
X = mat['TPRS']
with open('TPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('FPRS.mat')
X = mat['FPRS']
with open('FPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('RSE.mat')
X = mat['RSE']
with open('RSE.json', 'w') as FD:
    FD.write(json.dumps(X))


mat = io.loadmat('mu_TPRS.mat')
X = mat['TPRS']
with open('mu_TPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('mu_FPRS.mat')
X = mat['FPRS']
with open('mu_FPRS.json', 'w') as FD:
    FD.write(json.dumps(X))


mat = io.loadmat('sigma_TPRS.mat')
X = mat['TPRS']
with open('sigma_TPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('sigma_FPRS.mat')
X = mat['FPRS']
with open('sigma_FPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('ratio_TPRS.mat')
X = mat['TPRS']
with open('ratio_TPRS.json', 'w') as FD:
    FD.write(json.dumps(X))

mat = io.loadmat('ratio_FPRS.mat')
X = mat['FPRS']
with open('ratio_FPRS.json', 'w') as FD:
    FD.write(json.dumps(X))