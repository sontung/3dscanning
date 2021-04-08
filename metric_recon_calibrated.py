import numpy as np


def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    E = K2.T.dot(F).dot(K1)
    return E


def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s


def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    # TRIANGULATION
    # http://cmp.felk.cvut.cz/cmp/courses/TDV/2012W/lectures/tdv-2012-07-anot.pdf

    # Form of Triangulation :
    #
    # x = C.X
    #
    # |x|             | u |
    # |y| =   C(3x4). | v |
    # |1|             | w |
    #                 | 1 |
    #
    # 1 = C_3 . X
    #
    # x_i . (C_3_i.X_i) = C_1_i.X_i
    # y_i.  (C_3_i.X_i) = C_2_i.X_i

    # Subtract RHS from LHS and equate to 0
    # Take X common to get AX=0
    # Solve for X with SVD
    # for 2 points we have four equation

    P_i = []

    for i in range(pts1.shape[0]):
        A = np.array([   pts1[i,0]*C1[2,:] - C1[0,:] ,
                         pts1[i,1]*C1[2,:] - C1[1,:] ,
                         pts2[i,0]*C2[2,:] - C2[0,:] ,
                         pts2[i,1]*C2[2,:] - C2[1,:]   ])

        # print('A shape: ', A.shape)
        u, s, vh = np.linalg.svd(A)
        v = vh.T
        X = v[:,-1]
        # NORMALIZING
        X = X/X[-1]
        # print(X)
        P_i.append(X)

    P_i = np.asarray(P_i)

    # print('P_i: ', P_i)

    # MULTIPLYING TOGETHER WIH ALL ELEMENET OF Ps
    pts1_out = np.matmul(C1, P_i.T )
    pts2_out = np.matmul(C2, P_i.T )

    pts1_out = pts1_out.T
    pts2_out = pts2_out.T

    # NORMALIZING
    for i in range(pts1_out.shape[0]):
        pts1_out[i,:] = pts1_out[i,:] / pts1_out[i, -1]
        pts2_out[i,:] = pts2_out[i,:] / pts2_out[i, -1]

    # NON - HOMOGENIZING
    pts1_out = pts1_out[:, :-1]
    pts2_out = pts2_out[:, :-1]

    # print('pts2_out shape: ', pts2_out.shape)
    # print('pts1_out: ', pts1_out)
    # print('pts2_out: ', pts2_out)

    # CALCULATING REPROJECTION ERROR
    reprojection_err = 0
    for i in range(pts1_out.shape[0]):
        reprojection_err = reprojection_err  + np.linalg.norm( pts1[i,:] - pts1_out[i,:] )**2 + np.linalg.norm( pts2[i,:] - pts2_out[i,:] )**2

    # NON-HOMOGENIZING
    P_i = P_i[:, :-1]

    return P_i, reprojection_err


def bestM2(pts1, pts2, F, K1, K2):

    # CALCULATE E
    E = essentialMatrix(F, K1, K2)
    # CALCULATE M1 and M2
    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_list = camera2(E)

    #  TRIANGULATION
    C1 = K1.dot(M1)

    P_best = np.zeros( (pts1.shape[0],3) )
    M2_best = np.zeros( (3,4) )
    C2_best = np.zeros( (3,4) )
    err_best = np.inf

    error_list = []

    index = 0
    for i in range(M2_list.shape[2]):
        M2 = M2_list[:, :, i]
        C2 = K2.dot(M2)
        P_i, err = triangulate(C1, pts1, C2, pts2)
        error_list.append(err)
        z_list = P_i[:, 2]
        if all(z > 0 for z in z_list):
            index = i
            err_best = err
            P_best = P_i
            M2_best = M2
            C2_best = C2
    print('error_list: ', error_list)
    print('err_best: ', err_best)

    return P_best, C2_best, M2_best, err_best





