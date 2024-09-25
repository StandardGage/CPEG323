import armsim
from simple_cnn import simple_cnn
import sys


def main():
    if not sys.argv[1:]:
        print("Usage: python3 armsim_simple_cnn.py <image_file_path>")
        return

    image_file_name = sys.argv[1]
    with open('simple_cnn/simple_cnn.s', 'r') as f:
        armsim.parse(f.readlines())

    # Load the image into the memory of the assembly program
    image_matrix = simple_cnn.load_image(image_file_name)
    image_address = armsim.sym_table['image']
    for y in range(simple_cnn.INPUT_IMAGE_SIZE):
        pixel_address = image_address + y * simple_cnn.INPUT_IMAGE_SIZE
        for x in range(simple_cnn.INPUT_IMAGE_SIZE):
            pixel_address += x
            pixel = image_matrix[(y, x)]
            armsim.mem[pixel_address:pixel_address + 1] = pixel.to_bytes(1, byteorder='little', signed=False)

    armsim.run()

    # Print out the output of conv_max_pool
    print("Conv Max Pool Output:")
    output_address = armsim.sym_table['conv_max_pool_output']
    for k in range(simple_cnn.TOTAL_KERNELS):
        for j in range(simple_cnn.MAX_POOL_OUTPUT_SIZE):
            for i in range(simple_cnn.MAX_POOL_OUTPUT_SIZE):
                address = output_address
                address += ((k * simple_cnn.MAX_POOL_OUTPUT_SIZE + j) * simple_cnn.MAX_POOL_OUTPUT_SIZE + i) * 4
                value = int.from_bytes(bytes(armsim.mem[address:address + 4]), 'little', signed=True)
                print(value, end=" ")
            print()
        print()


if __name__ == "__main__":
    main()
