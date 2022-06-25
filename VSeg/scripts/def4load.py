import os
import torch
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

class Const_strcData:
    pass

def NM_DiArray(n,m):
    TMPS = []
    for k in range(n):
        TMP = []
        for j in range(m):
            column = []
            TMP.append(column)
        TMPS.append(TMP)
    return TMPS

def myload_DFKs_MAIN(INfile):
    # class Const_strcData:
    #     pass

    OUTT = Const_strcData()

    dataIO = loadmat(INfile)
    TMP = dataIO['DFKs']

    data_EIG = TMP['EIG']
    data_EIGLE = TMP['EIGLE']
    data_GRID = TMP['GRID']

    El1 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    El2 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    El3 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    Ev1 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    Ev2 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    Ev3 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T11 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T12 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T13 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T22 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T23 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    T33 = NM_DiArray(TMP.shape[0], TMP.shape[1])
    X = NM_DiArray(TMP.shape[0], TMP.shape[1])
    Y = NM_DiArray(TMP.shape[0], TMP.shape[1])
    Z = NM_DiArray(TMP.shape[0], TMP.shape[1])
    pad = NM_DiArray(TMP.shape[0], TMP.shape[1])

    for a0 in range(TMP.shape[0]):
        for a1 in range(TMP.shape[1]):
            El1[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['El1']).tolist())
            El2[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['El2']).tolist())
            El3[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['El3']).tolist())
            Ev1[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['Ev1']).tolist())
            Ev2[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['Ev2']).tolist())
            Ev3[a0][a1] = np.squeeze(np.array(data_EIG[a0][a1]['Ev3']).tolist())

            T11[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T11']).tolist())
            T12[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T12']).tolist())
            T13[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T13']).tolist())
            T22[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T22']).tolist())
            T23[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T23']).tolist())
            T33[a0][a1] = np.squeeze(np.array(data_EIGLE[a0][a1]['T33']).tolist())

            X[a0][a1] = np.squeeze(np.array(data_GRID[a0][a1]['X']).tolist())
            Y[a0][a1] = np.squeeze(np.array(data_GRID[a0][a1]['Y']).tolist())
            Z[a0][a1] = np.squeeze(np.array(data_GRID[a0][a1]['Z']).tolist())
            pad[a0][a1] = np.squeeze(np.array(data_GRID[a0][a1]['pad']).tolist())

    OUTT.k = TMP['k']
    OUTT.Dk = TMP['Dk']
    OUTT.phi = TMP['phi']
    OUTT.EIG_El1 = El1
    OUTT.EIG_El2 = El2
    OUTT.EIG_El3 = El3
    OUTT.EIG_Ev1 = Ev1
    OUTT.EIG_Ev2 = Ev2
    OUTT.EIG_Ev3 = Ev3
    OUTT.EIGLE_T11 = T11
    OUTT.EIGLE_T12 = T12
    OUTT.EIGLE_T13 = T13
    OUTT.EIGLE_T22 = T22
    OUTT.EIGLE_T23 = T23
    OUTT.EIGLE_T33 = T33
    OUTT.GRID_X = X
    OUTT.GRID_Y = Y
    OUTT.GRID_Z = Z
    OUTT.GRID_pad = pad

    return OUTT

def myload_PrincDirects(INfile):

    dataIO = loadmat(INfile)
    TMP = dataIO['PrincDirects']['M']

    OUTT = [None]*TMP.shape[1]
    for i in range(TMP.shape[1]):
        TMP2 = TMP[0][i]
        TMP2 = TMP2.astype(dtype='float32')
        TMP2 = torch.from_numpy(TMP2)
        OUTT[i] = TMP2#.to(device)

    return OUTT

def VTF3D_RotMatrix4Vect(x0, y0, x1, y1):
    # x0 = [x0,x0,x0]
    # x1 = [x1,x1,x1]
    tol = 1e-3

    if np.sum(np.abs(x0-x1)>tol) > 0: # if the inputs x0 and x1 are different
        if np.sum(np.abs(np.abs(x0)-np.abs(x1)) > tol) > 0: # if the input's absolute values are different (they are not just opposite -- sign)
            vX = np.cross(x1,x0) # we swopt x1 and x0, diff than Stefano's MATLAB
            sX = np.power(np.sum(np.power(vX, 2)), 0.5) # recheck in case of error, we got a slightest different results from MATLAB
            cX = np.dot(x0,x1)

            VXmat = np.zeros((3,3))
            VXmat[0, :] = [0, -vX[2], vX[1]]
            VXmat[1, :] = [vX[2], 0, -vX[0]]
            VXmat[2, :] = [-vX[1], vX[0], 0]

            R1 = np.eye(3) + VXmat + np.matmul(VXmat,VXmat)*((1-cX)/np.power(sX, 2))
        else: # in case x0 = -x1 , not yet check DANGEROUS
            R1 = eye(3)
            pos1 = np.array(np.where(x0*np.transpose(x1) < 0))
            for jj in range(pos1):
                R1[jj, jj] = -1
            #
    else:
        R1 = np.eye(3)

    # R2
    y0r = np.matmul(y0, R1)
    if np.sum(np.abs(y0r - y1) > tol) > 0:  # if the inputs y0r and y1 are different
        if np.sum(np.abs(np.abs(y0r) - np.abs(y1)) > tol) > 0:  # if the input's absolute values are different (they are not just opposite -- sign)
            vY = np.cross(y1,y0r) # we swopt x1 and x0, diff than Stefano's MATLAB
            sY = np.power(np.sum(np.power(vY, 2)), 0.5) # recheck in case of error, we got a slightest different results from MATLAB
            cY = np.dot(y0r,y1)

            VYmat = np.zeros((3,3))
            VYmat[0, :] = [0, -vY[2], vY[1]]
            VYmat[1, :] = [vY[2], 0, -vY[0]]
            VYmat[2, :] = [-vY[1], vY[0], 0]

            R2 = np.eye(3) + VYmat + np.matmul(VYmat, VYmat) * ((1 - cY) / np.power(sY, 2))
        else:
            R2 = eye(3)
            pos2 = np.array(np.where(y0r*np.transpose(y1) < 0))
            for kk in range(pos2):
                R2[kk, kk] = -1
    else:
        R2 = eye(3)

    R = np.matmul(R1, R2)
    return R

def VTF3D_Rot3DwithR(Img3D, R, ExtraValue):
    # Trasnlation
    dx = 0
    dy = 0
    dz = 0
    # Dimensions
    nY, nX, nZ = Img3D.shape[0], Img3D.shape[1], Img3D.shape[2] # Does nY and nX need to swopt???

    # Center Grids
    Y, X, Z = np.mgrid[0-np.round(nY/2):nY-np.round(nY/2), 0-np.round(nX/2):nX-np.round(nX/2), 0-np.round(nZ/2):nZ-np.round(nZ/2)]

    # Complete Transformation Matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0,3] = dx
    T[1,3] = dy
    T[2,3] = dz

    # Transformed Grids
    Xtr = -1 * T[0, 3] + T[0, 0] * X + T[0, 1] * Y + T[0, 2] * Z
    Ytr = -1 * T[1, 3] + T[1, 0] * X + T[1, 1] * Y + T[1, 2] * Z
    Ztr = -1 * T[2, 3] + T[2, 0] * X + T[2, 1] * Y + T[2, 2] * Z

    # % Interpolation (NEED TO check with stefano)
    # Img3DRotated = interpn(Y,X,Z,Img3D,Ytr,Xtr,Ztr,'linear');  % More Accurate but Slower
    # method='linear'
    # Img3DRotated(isnan(Img3DRotated)) = ExtraValue;

    #Interpolation method 1
    Y_chin = Y[:, 0, 0]
    X_chin = X[0, :, 0]
    Z_chin = Z[0, 0, :]
    Ytr_chin = Ytr[:, 0, 0]
    Xtr_chin = Xtr[0, :, 0]
    Ztr_chin = Ztr[0, 0, :]

    # Build data for the interpolator
    points_y, points_x, points_z = np.broadcast_arrays(Y_chin.reshape(-1, 1, 1), X_chin.reshape(1, -1, 1),
                                                       Z_chin.reshape(1, 1, -1))
    points = np.vstack((points_y.flatten(),
                        points_x.flatten(),
                        points_z.flatten()
                        )).T
    values = Img3D.flatten()

    f_3d_interp = LinearNDInterpolator(points, values)

    Img3DRotated = np.empty_like(Img3D)
    Img3DRotated = Img3DRotated * np.nan

    for yi in range(Img3DRotated.shape[0]):
        for xi in range(Img3DRotated.shape[1]):
            for zi in range(Img3DRotated.shape[2]):
                #print(yi, xi, zi)
                #print(Ytr[yi,xi,zi], Xtr[yi,xi,zi], Ztr[yi,xi,zi])
                Img3DRotated[yi, xi, zi] = f_3d_interp(np.array([[Ytr[yi,xi,zi], Xtr[yi,xi,zi], Ztr[yi,xi,zi]]]))
    Img3DRotated[np.isnan(Img3DRotated)] = ExtraValue

    # # Interpolation method 2
    # Y_chin = Y[:, 0, 0]
    # X_chin = X[0, :, 0]
    # Z_chin = Z[0, 0, :]
    # Ytr_chin = Ytr[:, 0, 0]
    # Xtr_chin = Xtr[0, :, 0]
    # Ztr_chin = Ztr[0, 0, :]
    #
    # fn_3d = RegularGridInterpolator((Y_chin, X_chin, Z_chin), Img3D)
    # Img3DRotated = np.empty_like(Img3D)
    # Img3DRotated = Img3DRotated * np.nan
    #
    # for yi in range(Img3DRotated.shape[0]):
    #     for xi in range(Img3DRotated.shape[1]):
    #         for zi in range(Img3DRotated.shape[2]):
    #             if (Ytr_chin[yi] <= Y_chin.max()) and (Ytr_chin[yi] >= Y_chin.min()):
    #                 if (Xtr_chin[xi] <= X_chin.max()) and (Xtr_chin[xi] >= X_chin.min()):
    #                     if (Ztr_chin[zi] <= Z_chin.max()) and (Ztr_chin[zi] >= Z_chin.min()):
    #                         #print("do sth for god sake")
    #                         Img3DRotated[yi, xi, zi] = fn_3d([[Ytr_chin[yi], Xtr_chin[xi], Ztr_chin[zi]]])
    #                     else:
    #                         Img3DRotated[yi, xi, zi] = ExtraValue
    #                 else:
    #                     Img3DRotated[yi, xi, zi] = ExtraValue
    #             else:
    #                 Img3DRotated[yi, xi, zi] = ExtraValue
    #
    # Img3DRotated[np.isnan(Img3DRotated)] = ExtraValue

    return Img3DRotated


def VTF3D_rotate3DKernel(Kernel, MainOrients, RotatedOrients):
    # Kernel 9x9x9
    # MainOrients 3x3
    # RotatedOrients 2x3
    R = VTF3D_RotMatrix4Vect(MainOrients[0,:].data.cpu().numpy(), MainOrients[1,:].data.cpu().numpy(), RotatedOrients[0,:].data.cpu().numpy(), RotatedOrients[1,:].data.cpu().numpy())
    # finish til now 20200625
    KernelRot = VTF3D_Rot3DwithR(Kernel.data.cpu().numpy(), R, np.nan)

    return KernelRot


