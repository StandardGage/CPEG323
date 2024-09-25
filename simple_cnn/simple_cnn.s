// ---------- SimpleCNN Model Parameters/Constants ----------
.data
// Kernels used for convolution. (6 x 5 x 5) signed bytes
conv_weights:   .byte 1, 2, 1, 1, 5, 2, 7, 4, 4, 4, -5, 1, 2, 0, 0, -9, -4, -4, -3, -2, 0, -3, -3, -2, 0, -4, 1, 3, 2, -1, -2, 3, 3, 1, -5, 5, 4, 2, 0, -4, 5, -1, -3, -2, -4, -5, -6, -1, -1, -2, 0, 3, 0, -3, 0, 8, 4, 2, -6, -6, -1, 1, 10, -2, -5, -8, -3, 9, 5, 1, -1, -4, 0, 4, 6, -3, 8, -2, -5, -3, 4, 5, -6, -5, -1, 4, 6, -3, 0, 1, 0, 1, 1, 0, 1, 1, 1, -1, 1, -1, -5, -3, 0, 4, 2, -2, 0, -1, 4, 4, -3, 0, -2, 3, 3, -3, -1, 0, 5, 2, -3, -4, -5, 0, 5, -7, -3, -2, -2, 4, -3, -4, -4, -6, -3, 3, -1, -1, -3, -5, 4, 3, 3, 1, -5, 3, 1, 2, 4, 2
// Biases used for convolution (6) signed bytes
conv_biases:    .byte -67, -114, -96, -54, -120, -128

// ---------- SimpleCNN Input/Outputs Matrices ----------
.bss
// Input image that is used by the convolution_max_pool procedure. (28 x 28) unsigned bytes
image:                  .space 784
// Temporary matrix used by the convolution_max_pool procedure to store the intermediate
// result of convolution. This is passed to the max_pool function. (24 x 24) signed ints
conv_output:            .space 2304
// This is used to store the result of the convolution_max_pool procedure
// This is passed to the max_pool function. (6 x 12 x 12) signed ints
conv_max_pool_output:   .space 3456

// ---------- Main Procedure (Non-Leaf) ----------
.text
.global _start
_start:
    LDUR X0, =image
    LDUR X1, =conv_weights
    LDUR X2, =conv_biases
    LDUR X3, =conv_max_pool_output
    BL convolution_max_pool
exit:
    // Exit sys call terminates program
    MOV X8, #93
    SVC 0

// ---------- ConvolutionMaxPool Procedure (Leaf) ----------
// Parameters:
//   X0: image
//   X1: weights
//   X2: biases
//   x3: output
convolution_max_pool:
    sub sp, sp, #32 
    stur lr, [sp, #0] 
    stur x19, [sp, #8]
    stur x20, [sp, #16]
    stur x21, [sp, #24]
    mov x19, x0      
                // x19 = image base address
    mov x20, x1      
                // x20 = weights base address
    mov x21, x2      
                // x21 = biases base address


    mov x0, #0 
                // k = 0
    loopk:
        cmp x0, #6
                    // k < TOTAL_KERNELS
        b.ge loopkdone
                    // finish k loop

        ldur x5, =conv_output
                    // x5 = conv_output

        mov x9, #0
                    // j = 0
        loopj:
            cmp x9, #24
                        // j < CONV_OUTPUT_SIZE
            b.ge loopjdone
                        // finish j loop

            mov x10, #0
                        // i = 0
            loopi:
                cmp x10, #24
                            // i < CONV_OUTPUT_SIZE
                b.ge loopidone
                            // finish i loop
                mov x4, #0
                            // sum = 0
                
                mov x11, #0
                            // y = 0
                loopy:
                    cmp x11, #5
                                // y < CONV_KERNEL_SIZE
                    b.ge loopydone
                                // finish y loop
                    
                    mov x12, #0
                                // x = 0
                    loopx:
                        cmp x12, #5
                                    // x < CONV_KERNEL_SIZE
                        b.ge loopxdone
                                    // finish x loop

                        add x13, x9, x11
                                    // x13 = j + y
                        add x14, x10, x12
                                    // x14 = i + x
                        mov x15, #28
                        //mul x13, x13, x15
                                    // multiply by 28 for 28x28 input
                        add x13, x13, x14
                                    // x13 = (j+y) * 28 + (i+x)
                        add x13, x19, x13
                                    // x13 = address of input[j+y][i+x]
                        ldur x13, [x13]
                                    // x13 = value of x13, input[j+y][i+x]
                        
                        mov x14, #25
                                    // 5x5 kernel
                        mul x14, x0, x14
                                    // x14 = k * 25
                        add x14, x14, x11
                                    // x14 = k * 25 + y
                        mov x15, #5
                        mul x14, x14, x15
                                    // x14 = (k * 25 + y) * 5
                        add x14, x14, x12
                                    // x14 = (k * 25 + y) * 5 + x
                        ldursb x14, [x20, x14]
                                    // load weights[k][y][x] into x14
                        mul x13, x13, x14
                                    // multiply input[j+y][i+x] * weights[k][y][x]
                        add x4, x4, x13
                                    // add to sum
                        add x12, x12, #1
                                    // x += 1
                        b loopx
                    loopxdone:
                        add x11, x11, #1
                                    // y += 1
                        b loopy
                loopydone:
                    add x10, x10, #1
                                // i += 1
                    mov x11, #24
                    mul x11, x9, x11
                                // x11 = j * 24
                    add x11, x11, x10
                                // x11 = j *24 + i
                    lsl x11, x11, #2
                                // x11 = (j*24+i) * 4
                    add x11, x5, x11
                                // x11 = conv_output[j][i]
                    add x12, x21, x0
                                // x12 = address of biases[k]
                    ldursb x6, [x12]
                                // x6 = biases[k]
                    add x6, x6, x4
                                // sum + biases[k]
                    mov x7, x0
                                // save k to x7
                    mov x0, x6
                                // move sum + biases[k] to x0
                    bl relu
                                // call relu
                    stur x0, [x11]
                                // store relu return in conv_output[j][i]
                    mov x0, x7
                                // restore k to x0
                    add x10, x10, #1
                                // i += 1
                    b loopi
            loopidone:
                add x9, x9, #1
                            // j += 1
                b loopj
            
    loopjdone:
        mov x1, x5
        mov x2, x3
        // make sure x0 = k, x1 = input, x2 = output
        //bl max_pool
        add x0, x0, #1
                    // k += 1
        b loopk
    
    loopkdone:
    // put stack back to initial state
    ldur lr, [sp, #0]
    ldur x19, [sp, #8]
    ldur x20, [sp, #16]
    ldur x21, [sp, #24]
    add sp, sp, #32
    BR LR

// ---------- MaxPool Procedure (Leaf) ----------
// Parameters:
//   X0: k (kernel index)
//   X1: input (base pointer to conv_output matrix)
//   X2: output (base pointer to conv_max_pool_output matrix)
max_pool:
    mov x9, #0
                // j = 0
    mloopj:
        cmp x9, #12
                // j < MAX_POOL_OUTPUT_SIZE
        b.ge mloopjdone
                // j loop done
        
        mov x10, #0
                // i = 0
        mloopi:
            cmp x10, #12
                    // i < MAX_POOL_OUTPUT_SIZE
            b.ge mloopidone
                    // i loop done
            
            mov x13, #0
                    // max = 0
            mov x11, #0
                    // y = 0
            mloopy:
                cmp x11, #2
                        // y < MAX_POOL_WINDOW_SIZE
                b.ge mloopydone
                        // y loop done
                
                mov x12, #0
                        // x = 0
                mloopx:
                    cmp x12, #2
                            // x < MAX_POOL_WINDOW_SIZE
                    b.ge mloopxdone
                            // x loop done
                    
                    lsl x14, x9, #1
                            // x14 = j * MAX_POOL_STRIDE (2)
                    add x14, x14, x11
                            // x14 = j * MAX_POOL_STRIDE + y
                    
                    lsl x15, x10, #1
                            // x15 = i * MAX_POOL_STRIDE (2)
                    add x15, x15, x12
                            // x15 = i * MAX_POOL_STRIDE + x
                    mov x16, #23
                    mul x14, x14, x16
                            // multiply by # of rows
                    add x14, x14, x15
                            // add offset
                    ldur x14, [x1, x14]
                            // get value of input
                    
                    cmp x14, x13
                            // compare input with max
                    b.gt replace
                    add x12, x12, #1
                    b mloopx

                    replace:
                        mov x13, x14
                        add x12, x12, #1
                        b mloopx
                mloopxdone:
                    add x11, x11, #1
                    b mloopy
            mloopydone:
                mov x11, #6
                mul x11, x11, x0
                        // x11 = k * 6
                mov x12, #12
                mul x12, x9, x12
                        // x12 = j * 12
                add x11, x11, x12
                add x11, x11, x10
                        // x11 = k + j + i (times row amounts)
                stur x13, [x2, x11]
                        // output[k][j][i] = max
                add x10, x10, #1
                        // i += 1
                b mloopi
        mloopidone:
            add x9, x9, #1
            b mloopj
    mloopjdone:
        BR LR

// ---------- ReLU Procedure (Leaf) ----------
// Parameters:
//   X0: x (convolution + bias)
// Returns:
//   X0: max(0, x)
relu:
    CMP X0, #0
    b.gt ret
    MOV X0, #0
    ret:
    BR LR
