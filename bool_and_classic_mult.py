import numpy as np


def standard_matrix_multiply(matrix_a, matrix_b):
    """
    Реализует стандартное умножение матриц по определению
    C_ij = sum_k A_ik * B_kj
    """
    assert matrix_a.shape[1] == matrix_b.shape[0], "Несовместимые размеры матриц для умножения"
    np_matrix1 = np.array(matrix_a)
    np_matrix2 = np.array(matrix_b)
    result = np.zeros((np_matrix1.shape[0], np_matrix2.shape[1]))
    for i in range(np_matrix1.shape[0]):
        for j in range(np_matrix2.shape[1]):
            for k in range(np_matrix2.shape[1]):
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return result.astype(int)


def boolean_matrix_multiply(matrix_a, matrix_b):
    """
    Реализует булево умножение матриц по определению
    C_ij = ⋁ (A_ik ∧ B_kj)
    """
    assert matrix_a.shape[1] == matrix_b.shape[0], "Несовместимые размеры матриц для умножения"
    np_matrix1 = np.array(matrix_a)
    np_matrix2 = np.array(matrix_b)
    result = np.zeros((np_matrix1.shape[0], np_matrix2.shape[1]), dtype=bool)
    for i in range(np_matrix1.shape[0]):
        for j in range(np_matrix2.shape[1]):
            for k in range(np_matrix2.shape[1]):
                if np_matrix1[i, k] and np_matrix2[k, j]:
                    result[i, j] = True
                    break
    return result.astype(int)


if __name__ == "__main__":
    matrix_с1 = np.array([[1, 2, 3],
                          [1, 2, 3],
                          [3, 4, 5]])
    matrix_с2 = np.array([[5, 6, 4],
                          [3, 4, 5],
                          [7, 8, 2]])

    print("Матрица A для стандартного умножения:")
    print(matrix_с1)

    print("\nМатрица B для стандартного умножения:")
    print(matrix_с2)

    print("\nСтандартное умножение")
    print(standard_matrix_multiply(matrix_с1, matrix_с2))

    matrix_b1 = np.array([[0, 1, 1, 1],
                          [0, 1, 0, 0],
                          [1, 1, 0, 1],
                          [1, 0, 0, 1]])

    matrix_b2 = np.array([[1, 0, 0, 1],
                          [0, 0, 1, 1],
                          [1, 0, 1, 0],
                          [0, 1, 0, 1]])

    print("\nМатрица A для бинарного умножения:")
    print(matrix_b1)

    print("\nМатрица B для бинарного умножения:")
    print(matrix_b2)

    print("\nБулево умножение:")
    print(boolean_matrix_multiply(matrix_b1, matrix_b2))
