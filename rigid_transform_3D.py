import numpy as np

'''
This function computes the optimal rigid transformation that best aligns two sets of data points. The transformation is calculated
in a weighted manner, where each point is associated with a weight, which should give each point a different importance. In this 
function, all the points have the same weight 1.

:param Matrix_A: matrix of dimension 3xN, this is the matrix of the target points, 3 for the three dimensions x, y and z and N points
:param Matrix_B: matrix of dimension 3xN this is the matrix of the source points, 3 for the three dimensions x, y and z and N points
:param weights: represents the weights of each point in the arrays A and B, which in this case will be 1

The function first centers both point clouds using w´the weighted centroids. Then it computes the weighted covariance matrix.
This covariance matrix is than used in the Singular Value Decomposition in order to find the optimal rotation matrix.
Reflection is also taken into consideration
Based on the corrected rotation and centroids, a translation vector is calculated

:return: 
The functions return a 3x3 rotation Matrix R, which represents the optimal rotation and a 1x3 translation vector t, representing 
the optimal translation
'''


def rigid_transform(matrix_A, matrix_B) -> tuple:


    # all elements of the list "weights" are given the same value, in this case 1

    num_rowsA, num_colsA = matrix_A.shape
    # this line assigns the first element of "Matrix_A" to "num_rowsA", which will than contain the number of rows
    # and the second element of "Matrix_A" to "num_colsA", which will than contain the columns of A
    if num_rowsA != 3:
        raise Exception(f"Matrix A is not a 3xN matrix")

    num_rowsB, num_colsB = matrix_B.shape
    if num_rowsB != 3:
        raise Exception(f"Matrix B is not a 3xN matrix")

    # the if checks wether both "Matrix_A" and "Matrix_B" are a 3xN matrix (3 because of the three coordinates x, y ans z
    # and N because we have N number of points)
    # centroidA = np.dot(matrix_A, weights)/np.sum(weights)
    # centroidB = np.dot(matrix_B, weights)/np.sum(weights)
    # calculation for the weighted centroids, this formula takes in consideration the values of each weight, so the weights
    # influence the contribution of each point to the centroid

    centroidA = np.mean(matrix_A, axis=1)
    centroidB = np.mean(matrix_B, axis=1)

    # As all weights have the same value 1, I simplified the formula in which I calculate the average of the coordinates along each
    # dimension (x, y and z). "axis=1" specifies that the mean should be calculated only from the rows(along each column) of the Matrix
    # "np.mean" is a NumPy function that calculates the mean/average of a specified axis.
    # CentroidA and centroidB are the centers of mass of matrix_A and matrix_B respectively

    centroidA = centroidA.reshape(-1, 1)
    centroidB = centroidB.reshape(-1, 1)

    # the centroids are an array with the shape (3,), so a 1D array. This line reshapes the centroids to a 2D Array with a shape (3, 1)

    Matrix_Am = matrix_A - centroidA
    Matrix_Bm = matrix_B - centroidB

    # This line is a centering operation which centers both "Matrix_A" and "Matrix_B" around their respective centroids.
    # So the importance is following:
    # 1. The subtraction moves the centroid to the origin of the coordinate system (0, 0, 0). This eases the geometric operations
    # such as rotation and scaling
    # 2. Centering both point clouds around their centroids creates a constant frame of reference making the alignment process easier
    # 3. It separates the translation from the alignment problem. Translational differences are isolated and finding the optimal
    # rotation in order to align the point clouds is easier

    # H = Am @ np.diag(weights) @ Bm.T

    # the multiplication with the diagonal matrix of the weights is unnecessary as it has the value 1

    Matrix_H = Matrix_Am @ Matrix_Bm.T

    # "@" is used for matrix multiplication
    # "Mareix_Bm" must be transposed as both matrix's are a 3xN matrix, so transposing "Matrix_B"m will make them compatible for
    # matrix multiplication
    # so instead of a 3xN * 3xN (which isn't possible), we now have a 3xN * Nx3 which will give us a 3x3 solution matrix
    # "Matrix_H" is a weighted covariance matrix which is needed in calculating the rotation between the two clouds
    # matrix_Am and matrix_Bm are the distances between each point and the center of mass of the two point clouds. They are needed,
    # so that we can align each center of mass to the center of the coordinate system(0,0,0). For example, if the center of mass has
    # the coordinates (3, 5, 7), and some point A has the coordinates (9, 7, 10), after aligning the point which is the center of the mass
    # with the origin of the coordinate system, the Point A has "new" coordinates which are (9-3, 7-5, 10-7) = (6, 2, 3)

    U, S, Vt = np.linalg.svd(Matrix_H)

    # "np.linalg.svd" is a NumPy function which invoces Singular Value Decomposition (SVD). SVD is used for matrix decomposition.
    # Because our "Matrix_H" is a 3x3 Matrix, the de decomposition leads to the formation of the 3 Vectros, which are respectively
    # assigned to "U", "S" and "Vt"
    # TODO: singular value decomposition

    R = Vt.T @ U.T

    # "R" is the optimal rotation matrix.
    # "U" captures how the SOURCE POINT CLOUD has to be rotated to allign with the target point cloud.
    # "Vt" captures how the TARGET POINT ClOUD has to be rotated to allign the source point cloud.
    # So by combining both of these vectors in "R", we obtain the optimal rotation matrix.
    # "S" contains the scaling factors, which aren't needed due to the multiplication of "U" and "Vt".
    # TODO: Linear transformation

    if np.linalg.det(R) < 0:
        print("reflection detected")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # This is a safety check, if the determinant of the "Matrix_A" is -1 it indicates reflection.
    # A reflection matrix represents mirroring or flipping of objects
    # In order to correct the reflection, we have to flip the signs of the z-component, which are the elements of the third row of "Vt"
    # After flipping the z-coordinates, "R" has to be rewritten/recomputed

    t = -R @ centroidA + centroidB

    # "t" is a translation vector which defines the amount and direction by which one point cloud needs to be translated to align
    # with the other
    # The "-" brings "centroidA" (the source point cloud) to its original orientation, while still being in the centered coord. system

    return R, t
