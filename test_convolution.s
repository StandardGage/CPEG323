.data
input:      .space 40000
weights:    .space 9
bias:       .space 1
output:     .space 40000

.text
.global _start
_start:
main:
    // ---------- Main Procedure ----------
    // x0 (n) will be set by armsim_convolution.py

// ---------- Convolution Procedure ----------
// Parameters:
// x0 = n
// x1 = &input: pointer to N x N matrix of signed words
// x2 = &weights: pointer to 3 x 3 matrix of signed bytes
// x3 = &bias: pointer to a single signed byte
// x4 = &output: pointer to (N - 2) x (N - 2) matrix of signed words
// Register Mapping:
// x19 = j
// x20 = i
// x23 = sum
// x24 = n
// x25 = &input
// x26 = &weights
// x27 = bias
// x28 = &output
// x5-x16, x8-x9, x10-x15: temporary registers (within allowed range)
    // Preserve LR and saved registers
    SUB SP, SP, #96
    STUR X19, [SP, #0]
    STUR X20, [SP, #8]
    STUR X23, [SP, #16]
    STUR X24, [SP, #24]
    STUR X25, [SP, #32]
    STUR X26, [SP, #40]
    STUR X27, [SP, #48]
    STUR X28, [SP, #56]
    STUR LR, [SP, #64]

    // Preserve Parameters
    MOV X24, X0           
    LDUR X25, =input          
    LDUR X26, =weights 
    // Load bias   
    LDUR X27, =bias   
    LDURSB X27, [X27]    
    LDUR X28, =output           
    

    // Start of convolution loops
convolution_loop_j:
    // exit if j >= n - 2
    SUB X8, X24, #2          
    // X8 = n - 2
    CMP X19, X8
    // i = 0
    MOV X20, XZR
    B.GE convolution_exit_loop_j

    // Precompute input_row_base = &input + j * n * 4
    MUL X5, X19, X24         
    // X5 = j * n
    LSL X5, X5, #2           
    // X5 = j * n * 4
    ADD X10, X25, X5         
    // X10 = input_row_base

    // Precompute output_row_base = &output + j * (n - 2) * 4
    SUB X8, X24, #2          
    // X8 = n - 2
    MUL X6, X19, X8          
    // X6 = j * (n - 2)
    LSL X6, X6, #2           
    // X6 = j * (n - 2) * 4
    ADD X11, X28, X6         
    // X11 = output_row_base

    // Precompute constants
    LSL X7, X24, #2          
    // X7 = n * 4
    ADD X16, X7, X7          
    // X16 = 2 * n * 4


convolution_loop_i:
    // exit if i >= n - 2
    SUB X8, X24, #2
    CMP X20, X8
    // Precompute input_element_base = input_row_base + i * 4
    LSL X8, X20, #2           
    B.GE convolution_exit_loop_i

    ADD X12, X10, X8          
    // X12 = input_element_base

    // Precompute output_addr = output_row_base + i * 4
    ADD X13, X11, X8          
    // X13 = output_addr

    // Initialize sum
    MOV X23, XZR

    // Unrolled convolution operation
    // Y = 0
    // X = 0
    LDURSW X8, [X12]
    LDURSB X9, [X26]
    MUL X23, X8, X9

    // X = 1
    LDURSW X8, [X12, #4]
    LDURSB X9, [X26, #1]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // X = 2
    LDURSW X8, [X12, #8]
    LDURSB X9, [X26, #2]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // Y = 1
    ADD X14, X12, X7          
    // X14 = input_element_base + n * 4
    // X = 0
    LDURSW X8, [X14]
    LDURSB X9, [X26, #3]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // X = 1
    LDURSW X8, [X14, #4]
    LDURSB X9, [X26, #4]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // X = 2
    LDURSW X8, [X14, #8]
    LDURSB X9, [X26, #5]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // Y = 2
    ADD X15, X12, X16         
    // X15 = input_element_base + 2 * n * 4
    // X = 0
    LDURSW X8, [X15]
    LDURSB X9, [X26, #6]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // X = 1
    LDURSW X8, [X15, #4]
    LDURSB X9, [X26, #7]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // X = 2
    LDURSW X8, [X15, #8]
    LDURSB X9, [X26, #8]
    MUL X8, X8, X9
    ADD X23, X23, X8

    // Add bias and apply ReLU
    ADD X23, X23, X27
    ASR X9, X23, #63          
    // Check if negative
    EOR X9, X9, X23
    AND X23, X9, X23          
    // Apply ReLU

    // Store the result
    STURW X23, [X13]

    // Increment i and loop
    ADD X20, X20, #1
    B convolution_loop_i

convolution_exit_loop_i:
    // Increment j and loop
    ADD X19, X19, #1
    B convolution_loop_j

convolution_exit_loop_j:
    // Restore LR and saved registers
    LDUR X19, [SP, #0]
    LDUR X20, [SP, #8]
    LDUR X23, [SP, #16]
    LDUR X24, [SP, #24]
    LDUR X25, [SP, #32]
    LDUR X26, [SP, #40]
    LDUR X27, [SP, #48]
    LDUR X28, [SP, #56]
    LDUR LR, [SP, #64]
    ADD SP, SP, #96

// Exit the program
    MOV X8, #93
    SVC 0