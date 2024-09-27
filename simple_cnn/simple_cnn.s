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
// result of convolution. This is passed to the max_pool function. (24 x 24) signed ints (4 bytes)
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
    sub sp, sp, #48
    // allocate space for lr and parameters
    stur lr, [sp, #0] 
    // save lr
    stur x19, [sp, #8]
    // save x19
    stur x20, [sp, #16]
    // save x20
    stur x21, [sp, #24]
    // save x21
    stur x22, [sp, #32]
    // save x22
    mov x19, x0      
                // x19 = image base address
    mov x20, x1      
                // x20 = weights base address
    mov x21, x2      
                // x21 = biases base address
    mov x22, x3
                // x22 = output


    mov x3, #0 
                // k = 0
    loopk:
        cmp x3, #6
                // k < TOTAL_KERNELS
        b.ge loopkdone
                // end k loop

        ldur x9, =conv_output
                // x9 = convolution_output

        mov x4, #0
        // j = 0
        loopj:
                cmp x4, #24
                // j < CONV_OUTPUT_SIZE
                b.ge loopjdone
                // end j loop

                mov x5, #0
                // i = 0
                loopi:
                        cmp x5, #24
                        // i < CONV_OUTPUT_SIZE
                        b.ge loopidone
                        // end i loop

                        mov x10, #0
                        // sum = 0
                        mov x6, #0
                        // y = 0
                        loopy:
                            cmp x6, #5
                            // y < CONV_KERNEL_SIZE
                            b.ge loopydone
                            // end y loop

                                mov x7, #0
                                // x = 0
                                loopx:
                                    cmp x7, #5
                                    // x < CONV_KERNEL_SIZE
                                    b.ge loopxdone
                                    // end x loop

                                    add x11, x4, x6
                                    // x11 = j+y
                                    mov x12, #28
                                    // x12 = 28 (for # of rows)
                                    mul x11, x11, x12
                                    // x11 = 28*(j+y)
                                    add x12, x5, x7
                                    // x12 = i + x
                                    add x11, x11, x12
                                    // x11 = 28*(j+y)+(i+x)
                                    ldurb x11, [x19, x11]
                                    // x11 = input[j+y][i+x]

                                    mov x12, #25
                                    // x12 = 25 (for # of boxes)
                                    mul x12, x12, x3
                                    // x12 = 25 * k
                                    mov x13, #5
                                    // x13 = 5 (for # of rows)
                                    mul x13 , x13, x6
                                    // x13 = 5 * y
                                    add x12, x12, x13
                                    // x12 = (25*k)+(5*y)
                                    add x12, x12, x7
                                    // x12 = (25*k)+(5*y)+x
                                    ldursb x12, [x20, x12]
                                    // x12 = weights[k][y][x]

                                    mul x11, x11, x12
                                    // x11 = input[j+y][i+x] * weights[k][y][x]
                                    add x10, x10, x11
                                    // x10 += x11


                                    add x7, x7, #1
                                    // x += 1
                                    b loopx
                                    // continue loop x

                            loopxdone:
                            add x6, x6, #1
                            // y += 1
                            b loopy
                            // continue loop y
                        
                        loopydone:
                        ldursb x0, [x21, x3]
                        // x0 = biases[k]
                        add x0, x0, x10
                        // x0 = biases[k] + sum
                        bl relu
                        // call relu(biases[k] + sum)
                        mov x11, #24
                        // x11 = 24 (for # of rows)
                        mul x11, x11, x4
                        // x11 = 24 * j
                        add x11, x11, x5
                        // x11 = 24 * j + i
                        lsl x11, x11, #2
                        // x11 = (24*j + i) * 4
                        sturw x0, [x9, x11]
                        // conv_output[j][i] = relu(sum + biases[k])

                        add x5, x5, #1
                        // i += 1
                        b loopi
                        // continue loop i

                loopidone:
                add x4, x4, #1
                // j += 1
                b loopj
                // continue loop j

        loopjdone:

        // prepare to call max_pool
        mov x0, x3
        // mov k to x0
        mov x1, x9
        // mov conv_output to x1
        mov x2, x22
        // mov output to x2
        bl max_pool
        // call max_pool

        add x3, x3, #1
        // k += 1
        b loopk
        // continue loop k
    
    loopkdone:
    // put stack back to initial state
    ldur lr, [sp, #0]
    // load original lr
    ldur x19, [sp, #8]
    // load original x19
    ldur x20, [sp, #16]
    // load original x20
    ldur x21, [sp, #24]
    // load original x21
    ldur x22, [sp, #32]
    // load original x22
    add sp, sp, #48
    // dealocate stack
    BR LR

// ---------- MaxPool Procedure (Leaf) ----------
// Parameters:
//   X0: k (kernel index)
//   X1: input (base pointer to conv_output matrix)
//   X2: output (base pointer to conv_max_pool_output matrix)
max_pool:
    mov x4, #0
    // j = 0
    mloopj:
        cmp x4, #12
        // j < MAX_POOL_OUTPUT_SIZE
        b.ge endmloopj
        // end loop j

        mov x5, #0
        // i = 0
        mloopi:
            cmp x5, #12
            // i < MAX_POOL_OUTPUT_SIZE
            b.ge endmloopi
            // end loop j

            mov x9, #0
            // max = 0
            mov x6, #0
            // y = 0
            mloopy:
                cmp x6, #2
                // y < MAX_POOL_WINDOW_SIZE
                b.ge endmloopy
                // end loop y

                mov x7, #0
                // x = 0
                mloopx:
                    cmp x7, #2
                    // x < MAX_POOL_WINDOW_SIZE
                    b.ge endmloopx
                    // end loop x

                    lsl x11, x4, #1
                    // x11 = j * 2
                    add x11, x11, x6
                    // x11 = j * 2 + y
                    mov x12, #24
                    // x12 = 24 (for # of rows)
                    mul x11, x11, x12 
                    // x11 = (j * 2 + y) * 24
                    lsl x12, x5, #1
                    // x12 = i * 2
                    add x12, x12, x7
                    // x12 = i * 2 + x
                    add x11, x11, x12
                    // x11 = 24*(j*2+y)+(i*2+x)
                    lsl x11, x11, #2
                    // x11 = (24*(j*2+y)+(i*2+x))*4
                    ldursw x10, [x1, x11]
                    // x10 = input[x11]
                    cmp x10, x9
                    // x10 > max ?
                    b.gt replace
                    // if it is, replace max with x10


                    add x7, x7, #1
                    // x += 1
                    b mloopx
                    // continue loop x

                    replace:
                        mov x9, x10
                        // replace x9 with x10
                        add x7, x7, #1
                        // x += 1
                        b mloopx
                        // continue loop x

                endmloopx:
                add x6, x6, #1
                // y += 1
                b mloopy
                // continue loop y

            endmloopy:
            mov x11, #144
            // x11 = 144 (because each block is 12x12)
            mul x11, x11, x0
            // x11 = 144 * k
            mov x12, #12
            // x12 = 12 (for # of rows)
            mul x12, x12, x4
            // x12 = 12 * j
            add x11, x11, x12
            // x11 = 144*k + 12*j
            add x11, x11, x5
            // x11 = 144*k + 12*j + i
            lsl x11, x11, #2
            // x11 = (144*k + 12*j + i) * 4
            sturw x9, [x2, x11]
            // output[k][j][i] = max

            add x5, x5, #1
            // i += 1
            b mloopi
            // continue loop i

        endmloopi:
        add x4, x4, #1
        // j += 1
        b mloopj
        // continue loop j
        
    endmloopj:
    BR LR

// ---------- ReLU Procedure (Leaf) ----------
// Parameters:
//   X0: x (convolution + bias)
// Returns:
//   X0: max(0, x)
relu:
    CMP X0, #0
    // x0 < 0 ?
    b.gt ret
    // if its not return
    MOV X0, #0
    // if it is, set x0 to 0
    ret:
    BR LR
