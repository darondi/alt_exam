import numpy as np


def split_to_2x2_blocks(matrix):
    """
    Разбивает квадратную матрицу размера n x n (n — чётное) на 4 блока размера (n/2)x(n/2)
    Возвращает матрицы в виде:
    [[A_11, A_12],
    [A_21, A_22]]
    """
    blocks = np.vsplit(matrix, 2)
    return [np.hsplit(block, 2) for block in blocks]


def strassen_mul_blocks(a, b):
    """
    Вычисляет произведение двух блочных матриц по алгоритму Штрассена,
    а именно сначала 7 рекурсивных произведений, затем собирает результирующие блоки
    a и b — это блочные матрицы вида:
    [[A_11, A_12],
    [A_21, A_22]]
    """
    d = strassen_mul(a[0][0] + a[1][1], b[0][0] + b[1][1])  # (A_11 + A_22)(B_11 + B_22)
    d_1 = strassen_mul(a[0][1] - a[1][1], b[1][0] + b[1][1])  # (A_12 - A_22)(B_21 + B_22)
    d_2 = strassen_mul(a[1][0] - a[0][0], b[0][0] + b[0][1])  # (A_21 - A_11)(B_11 + B_12)
    h_1 = strassen_mul(a[0][0] + a[0][1], b[1][1])  # (A_11 + A_12) B_22
    h_2 = strassen_mul(a[1][0] + a[1][1], b[0][0])  # (A_21 + A_22) B_11
    v_1 = strassen_mul(a[1][1], b[1][0] - b[0][0])  # A_22 (B_21 - B_11)
    v_2 = strassen_mul(a[0][0], b[0][1] - b[1][1])  # A_11 (B_12 - B_22)

    c_11 = d + d_1 + v_1 - h_1
    c_12 = v_2 + h_1
    c_21 = v_1 + h_2
    c_22 = d + d_2 + v_2 - h_2

    return [
        [c_11, c_12],
        [c_21, c_22]
    ]


def next_power_of_two(n):
    """ Возвращает наименьшую степень двойки, большую или равную n """
    return 1 << (n - 1).bit_length()


def add_to_power_of_two(matrix):
    """ Дополняет матрицу нулями до ближайшей степени двойки """
    n = matrix.shape[0]
    new_size = next_power_of_two(n)
    if n == new_size:
        return matrix.copy()
    padded = np.zeros((new_size, new_size))
    padded[:n, :n] = matrix
    return padded


def strassen_mul(left, right):
    """ Основная функция умножения матриц по алгоритму Штрассена """
    assert left.shape == right.shape, "Матрицы должны быть одинакового размера"
    assert left.shape[0] == left.shape[1], "Матрицы должны быть квадратными"

    n = left.shape[0]
    # Базовый случай, когда матрица имеет размер 1х1
    if n == 1:
        return np.array([[left[0, 0] * right[0, 0]]])
    assert n % 2 == 0, "Размер матрицы должен быть степенью двойки для разделения на блоки"

    # Разбиение обеих матриц на 4 блока
    a_blocks = split_to_2x2_blocks(left)
    b_blocks = split_to_2x2_blocks(right)

    c_blocks = strassen_mul_blocks(a_blocks, b_blocks)
    return np.block(c_blocks)


def strassen_multiply(matrix_a, matrix_b):
    """ Результирующая функция, которая также обрабатывает матрицы с размером не равным степени двойки """
    assert matrix_a.shape[1] == matrix_b.shape[0], "Несовместимые размеры матриц для умножения"

    original_rows = matrix_a.shape[0]
    original_cols = matrix_b.shape[1]

    add_a = add_to_power_of_two(matrix_a)
    add_b = add_to_power_of_two(matrix_b)

    result = strassen_mul(add_a, add_b)
    return result[:original_rows, :original_cols]


if __name__ == "__main__":
    matrix_a = np.array([[1, 2],
                         [3, 4]])
    matrix_b = np.array([[5, 6],
                         [7, 8]])

    print("Матрица A:")
    print(matrix_a)

    print("\nМатрица B:")
    print(matrix_a)

    C = strassen_multiply(matrix_a, matrix_b)
    print("\nРезультат умножения AB:")
    print(C)
    expected = np.dot(matrix_a, matrix_b)
    print("\nПроверка корректности алгоритма:")
    print(expected)

    print("\nКорректность алгоритма:", np.allclose(C, expected))
