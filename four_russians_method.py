import math


def build_table(k):
    """
    Создает таблицу предвычисленных результатов логического ИЛИ от всех комбинаций:
        result = OR (A_block AND B_block)
    """
    lookup = []
    for i in range(2 ** k):  # перебираем все возможные маски длины k
        bits_i = [(i >> j) & 1 for j in range(k)]  # битовая маска
        row_table = []
        for j in range((2 ** k)):  # перебираем все возможные значения второго блока
            bits_j = [(j >> jj) & 1 for jj in range(k)]
            res = 0
            for b1, b2 in zip(bits_i, bits_j):
                res |= (b1 & b2)  # логическое AND + OR
            row_table.append(res)
        lookup.append(row_table)
    return lookup


def boolean_matrix_multiply_four_russians(input_a, input_b):
    """
    Реализует булево умножение матриц методом "Четыре русских"
    C_ij = ⋁ (A_ik ∧ B_kj)
    """
    n = len(input_a)
    t = int(math.log2(n)) + 1  # размер блока t = log2(n)
    if t > n:
        t = n
    num_blocks = (n + t - 1) // t
    blocks_a = []
    for col_start in range(0, n, t):
        col_end = min(col_start + t, n)
        compressed_a = []
        for i in range(n):
            mask = 0
            for j in range(col_start, col_end):
                mask = (mask << 1) | input_a[i][j]
            compressed_a.append(mask)
        lookup = build_table(col_end - col_start)
        blocks_a.append((compressed_a, lookup))

    output = [[0] * n for _ in range(n)]
    for i in range(n):
        for block_idx in range(num_blocks):
            col_start = block_idx * t
            col_end = min(col_start + t, n)
            compressed_a_row = blocks_a[block_idx][0][i]
            lookup_table = blocks_a[block_idx][1]
            for j in range(n):
                # Формирование маски для соответствующей строки B
                mask_b = 0
                for jj in range(col_start, col_end):
                    mask_b = (mask_b << 1) | input_b[jj][j]
                val = lookup_table[compressed_a_row][mask_b]
                output[i][j] |= val # Применение OR
    return output


if __name__ == "__main__":
    matrix_a = [
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 1]
    ]

    matrix_b = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]

    result = boolean_matrix_multiply_four_russians(matrix_a, matrix_b)
    print("Результат умножения:")
    for row in result:
        print(row)