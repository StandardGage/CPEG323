import armsim
import random
import sys


def main():
    if not sys.argv[2:]:
        print("Usage: python3 armsim_matmul.py <assembly_file_path> <n>")
        return

    file_name = sys.argv[1]
    n = int(sys.argv[2])

    if n <= 0 or n > 100:
        print("n must be between 1-100")
        return

    with open(file_name, 'r') as f:
        armsim.parse(f.readlines())

    matrix_a_address = armsim.sym_table['matrixa']
    randomize_matrix(matrix_a_address, n)
    print_matrix('Matrix A', get_matrix(matrix_a_address, n))
    armsim.reg['x0'] = matrix_a_address

    matrix_b_address = armsim.sym_table['matrixb']
    randomize_matrix(matrix_b_address, n)
    print_matrix('Matrix B', get_matrix(matrix_b_address, n))
    armsim.reg['x1'] = matrix_b_address

    matrix_c_address = armsim.sym_table['matrixc']
    armsim.reg['x2'] = matrix_c_address
    armsim.reg['x3'] = n
    armsim.run()

    actual_values = get_matrix(matrix_c_address, n)
    print_matrix('Actual Matrix C', actual_values)

    expected_values = matrix_mul(matrix_a_address, matrix_b_address, n)
    print_matrix('Expected Matrix C', expected_values)

    print(f"Instruction Count: {armsim.execute_count}")
    print(f"Cycle Count: {armsim.cycle_count}")

    assert actual_values == expected_values


def randomize_matrix(matrix_address: int, n: int, min_value: int = 0, max_value: int = 10):
    length = n * n * 8
    for offset in range(0, length, 8):
        element_address = matrix_address + offset
        value = random.randint(min_value, max_value)
        armsim.mem[element_address:element_address + 8] = value.to_bytes(8, byteorder='little')


def print_matrix(label: str, matrix: list[list[int]]):
    print(label)
    for row in matrix:
        print(row)
    print()


def get_matrix(matrix_address: int, n: int) -> list[list[int]]:
    matrix = [[]] * n
    for i in range(n):
        row = [0] * n
        row_offset = i * n * 8
        for j in range(n):
            element_address = matrix_address + row_offset + j * 8
            row[j] = int.from_bytes(armsim.mem[element_address:element_address + 8], byteorder='little')
        matrix[i] = row
    return matrix


def matrix_mul(matrix_a_address: int, matrix_b_address: int, n: int) -> list[list[int]]:
    matrix = [[]] * n
    for i in range(n):
        row = [0] * n
        for j in range(n):
            sum_ = 0
            for k in range(n):
                a_i_k_address = matrix_a_address + (i * n * 8) + (k * 8)
                a_i_k_value = int.from_bytes(armsim.mem[a_i_k_address:a_i_k_address + 8], byteorder='little')
                b_k_j_address = matrix_b_address + (k * n * 8) + (j * 8)
                b_k_j_value = int.from_bytes(armsim.mem[b_k_j_address:b_k_j_address + 8], byteorder='little')
                sum_ += a_i_k_value * b_k_j_value
            row[j] = sum_
        matrix[i] = row
    return matrix


if __name__ == '__main__':
    main()
