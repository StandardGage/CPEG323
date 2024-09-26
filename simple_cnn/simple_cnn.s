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
    stur lr, [sp, #0] 
    stur x19, [sp, #8]
    stur x20, [sp, #16]
    stur x21, [sp, #24]
    stur x22, [sp, #32]
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
        b.ge loopkdone

        ldur x9, =conv_output
        mov x4, #0
        loopj:
                cmp x4, #24
                b.ge loopjdone

                mov x5, #0
                loopi:
                        cmp x5, #24
                        b.ge loopidone

                        mov x10, #0
                        // sum = 0
                        mov x6, #0
                        loopy:
                            cmp x6, #5
                            b.ge loopydone

                                mov x7, #0
                                loopx:
                                    cmp x7, #5
                                    b.ge loopxdone

                                    add x11, x4, x6
                                    mov x12, #28
                                    mul x11, x11, x12
                                    // x11 = 24*(j+y)
                                    add x12, x5, x7
                                    // x12 = i + x
                                    add x11, x11, x12
                                    ldurb x11, [x19, x11]

                                    mov x12, #25
                                    mul x12, x12, x3
                                    mov x13, #5
                                    mul x13 , x13, x6
                                    add x12, x12, x13
                                    add x12, x12, x7
                                    lsl x12, x12, #2
                                    ldurb x12, [x20, x12]

                                    mul x11, x11, x12
                                    add x10, x10, x11


                                    add x7, x7, #1
                                    b loopx

                            loopxdone:
                            add x6, x6, #1
                            b loopy
                        
                        loopydone:
                        lsl x0, x3, #2
                        ldurb x0, [x21, x11]
                        // x0 = biases[k]
                        add x0, x0, x10
                        bl relu
                        mov x11, #24
                        mul x11, x11, x4
                        add x11, x11, x5
                        lsl x11, x11, #2
                        // x11 = (24*j + i) * 4
                        sturb x0, [x9, x11]

                        add x5, x5, #1
                        b loopi

                loopidone:
                add x4, x4, #1
                b loopj

        loopjdone:

        mov x0, x3
        mov x1, x9
        mov x2, x22
        bl max_pool

        add x3, x3, #1
        b loopk
    
    loopkdone:
    // put stack back to initial state
    ldur lr, [sp, #0]
    ldur x19, [sp, #8]
    ldur x20, [sp, #16]
    ldur x21, [sp, #24]
    ldur x22, [sp, #32]
    add sp, sp, #48
    BR LR

// ---------- MaxPool Procedure (Leaf) ----------
// Parameters:
//   X0: k (kernel index)
//   X1: input (base pointer to conv_output matrix)
//   X2: output (base pointer to conv_max_pool_output matrix)
max_pool:
    mov x4, #0
    mloopj:
        cmp x4, #12
        b.ge endmloopj

        mov x5, #0
        mloopi:
            cmp x5, #12
            b.ge endmloopi

            mov x9, #0
            // max = 0
            mov x6, #0
            mloopy:
                cmp x6, #2
                b.ge endmloopy

                mov x7, #0
                mloopx:
                    cmp x7, #2
                    b.ge endmloopx

                    lsl x11, x4, #1
                    add x11, x11, x6
                    mov x12, #24
                    mul x11, x11, x12 
                    // x11 = (j * 2 + y) * 24
                    lsl x12, x5, #1
                    add x12, x12, x7
                    add x11, x11, x12
                    lsl x11, x11, #2
                    // x11 = (24*(j*2+y)+(i*2+y))*4
                    ldurb x10, [x1, x11]
                    // x10 = input[x11]
                    cmp x10, x9
                    b.gt replace


                    add x7, x7, #1
                    b mloopx

                    replace:
                        mov x9, x10
                        add x7, x7, #1
                        b mloopx

                endmloopx:
                add x6, x6, #1
                b mloopy

            endmloopy:
            mov x11, #144
            mul x11, x11, x0
            mov x12, #12
            mul x12, x12, x4
            add x11, x11, x12
            add x11, x11, x5
            lsl x11, x11, #2
            sturb x9, [x2, x11]

            add x5, x5, #1
            b mloopi

        endmloopi:
        add x4, x4, #1
        b mloopj
        
    endmloopj:
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
