.data
input:		.space 40000
weights:	.space 9
bias:		.space 1
output:		.space 40000

.text
.global _start
_start:
main:
// ---------- Main Procedure ----------
// x0 (n) will be set by armsim_convolution.py
LDUR X1, =input
LDUR X2, =weights
LDUR X3, =bias
LDUR X4, =output
BL convolution
exit:
// Exit sys call terminates program
MOV X8, #93
SVC 0

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
// x21 = y
// x22 = x
// x23 = sum
// x24 = n
// x25 = &input
// x26 = &weights
// x27 = &bias
// x28 = &output
convolution:
// Preserve LR and saved registers
SUB SP, SP, #96
STUR X19, [SP, #0]
STUR X20, [SP, #8]
STUR X21, [SP, #16]
STUR X22, [SP, #24]
STUR X23, [SP, #32]
STUR X24, [SP, #40]
STUR X25, [SP, #48]
STUR X26, [SP, #56]
STUR X27, [SP, #64]
STUR X28, [SP, #72]
STUR LR, [SP, #80]

// Preserve Parameters
MOV X24, X0
MOV X25, X1
MOV X26, X2
MOV X27, X3
MOV X28, X4

// j = 0
MOV X19, XZR
convolution_loop_j:
// exit if j >= n - 2
SUB X8, X24, #2
CMP X19, X8
B.GE convolution_exit_loop_j

// i = 0
MOV X20, XZR
convolution_loop_i:
// exit if i >= n - 2
SUB X8, X24, #2
CMP X20, X8
B.GE convolution_exit_loop_i

// y = 0, sum = 0
MOV X21, XZR
MOV X23, XZR
convolution_loop_y:
// exit if y >= 3
CMP X21, #3
B.GE convolution_exit_loop_y

// x = 0
MOV X22, XZR
convolution_loop_x:
// exit if x >= 3
CMP X22, #3
B.GE convolution_exit_loop_x

// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
ADD X8, X19, X21
MUL X8, X8, X24
ADD X8, X8, X20
ADD X8, X8, X22
MOV X9, #4
MUL X8, X8, X9
ADD X8, X8, X25
LDURSW X8, [X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
MOV X9, #3
MUL X9, X21, X9
ADD X9, X9, X22
ADD X9, X26, X9
LDURSB X9, [X9]
// sum += input[j + y][i + x] * weights[y][x]
MUL X8, X8, X9
ADD X23, X23, X8
//  x++
ADD X22, X22, #1
B convolution_loop_x

convolution_exit_loop_x:
// y++
ADD X21, X21, #1
B convolution_loop_y

convolution_exit_loop_y:
// sum += *bias
LDURSB X8, [X27]
ADD X23, X23, X8
// x0 = relu(sum)
MOV X0, X23
BL relu
// x8 = &output + (j * (n - 2) + i) * 4
SUB X8, X24, #2
MUL X8, X8, X19
ADD X8, X8, X20
MOV X9, #4
MUL X8, X8, X9
ADD X8, X28, X8
// output[j][i] = sum
STURW X0, [X8]
// i++
ADD X20, X20, #1
B convolution_loop_i

convolution_exit_loop_i:

// j++
ADD X19, X19, #1
B convolution_loop_j

convolution_exit_loop_j:
// Restore LR and saved registers
LDUR X19, [SP, #0]
LDUR X20, [SP, #8]
LDUR X21, [SP, #16]
LDUR X22, [SP, #24]
LDUR X23, [SP, #32]
LDUR X24, [SP, #40]
LDUR X25, [SP, #48]
LDUR X26, [SP, #56]
LDUR X27, [SP, #64]
LDUR X28, [SP, #72]
LDUR LR, [SP, #80]
ADD SP, SP, #96
BR LR

// ---------- RELU Procedure ----------
// Parameters:
// x0 = x
// Returns:
// x0 = max(0, x)
relu:
CMP X0, #0
B.GE relu_return
MOV X0, XZR
relu_return:
BR LR