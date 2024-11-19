import argparse
import armdb
import armsim
import random


def iterate_matrix(dimensions, current_index=None, depth=0):
    if current_index is None:
        current_index = [0] * len(dimensions)

    if depth == len(dimensions):
        yield tuple(current_index)
    else:
        for i in range(dimensions[depth]):
            current_index[depth] = i
            yield from iterate_matrix(dimensions, current_index, depth + 1)


def create_matrix(dimensions):
    return {index: 0 for index in iterate_matrix(dimensions)}


def set_matrix(matrix, dimensions, values):
    if len(matrix) != len(values):
        raise ValueError(f"Invalid number of values for given matrix: expected {len(matrix)} but got {len(values)}")

    for i, index in enumerate(iterate_matrix(dimensions)):
        matrix[index] = values[i]

def convolution(n, _input, weights, bias):
    output = create_matrix(((n - 2), (n - 2)))
    for j in range(n - 2):
        for i in range (n - 2):
            _sum = 0
            for y in range(3):
                for x in range(3):
                    _sum += _input[(j + y, i + x)] * weights[(y, x)]
            output[(j, i)] = relu(_sum + bias[(0,)])
    return output

def relu(x):
    return max(0, x)

def create_random_parameters(n):
    _input = create_matrix((n, n))
    set_matrix(_input, (n, n), [random.randint(0, 1000) for _ in range(n * n)])
    weights = create_matrix((3, 3))
    set_matrix(weights, (3, 3), [random.randint(-128, 127) for _ in range(9)])
    bias = create_matrix((1,))
    set_matrix(bias, (1,), [random.randint(-128, 127)])
    return _input, weights, bias

def armsim_inject_parameters(n, _input, weights, bias):
    armsim.reg['x0'] = n

    input_address = armsim.sym_table['input']
    for index in iterate_matrix((n, n)):
        y, x = index
        address = input_address + (y * n + x) * 4
        value = _input[(y, x)]
        armsim.mem[address:address + 4] = value.to_bytes(4, byteorder='little', signed=True)

    weights_address = armsim.sym_table['weights']
    for index in iterate_matrix((3, 3)):
        y, x = index
        address = weights_address + y * 3 + x
        value = weights[(y, x)]
        armsim.mem[address:address + 1] = value.to_bytes(1, byteorder='little', signed=True)

    bias_address = armsim.sym_table['bias']
    armsim.mem[bias_address:bias_address + 1] = bias[(0,)].to_bytes(1, byteorder='little', signed=True)

def armsim_extract_output(n):
    output = create_matrix((n - 2, n - 2))
    output_address = armsim.sym_table['output']
    for index in iterate_matrix((n - 2, n - 2)):
        y, x = index
        address = output_address + (y * (n - 2) + x) * 4
        value = int.from_bytes(bytes(armsim.mem[address:address + 4]), 'little', signed=True)
        output[(y, x)] = value
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Process convolution of N x N matrix with 3x3 kernel.")
    parser.add_argument(
        "n",
        type=int,
        choices=range(3, 101),
        help="input matrix size (integer in the range 3 to 100)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="enable debug mode"
    )

    return parser.parse_args()

def main():
    random.seed(0)
    args = parse_args()

    n = args.n
    _input, weights, bias = create_random_parameters(n)
    if args.debug:
        def init():
            armsim_inject_parameters(n, _input, weights, bias)
        armdb.main('convolution.s', init)
    else:
        with open('convolution.s', 'r') as f:
            armsim.parse(f.readlines())
            armsim_inject_parameters(n, _input, weights, bias)
            armsim.run()

    actual_output = armsim_extract_output(n)
    expected_output = convolution(n, _input, weights, bias)
    correct = actual_output == expected_output

    print(f"Total Cycles: {armsim.cycle_count}")
    print(f"Executed Instructions: {armsim.execute_count}")
    print("Assembly Code Correctness:", "Passed" if correct else "Failed")

if __name__ == "__main__":
    main()
