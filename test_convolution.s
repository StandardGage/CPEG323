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
//convolution:
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

// i and j always run at least once (so get that done without branching)

// -------------------Y == 0-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
// initial input values will be 0
LDURSW X8, [X25]
LDURSB X9, [X26]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X23, X8, X9
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LDURSW X8, [X25, #4]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [x26, #1]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LDURSW X8, [X25, #8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #2]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 0-------------------

// -------------------Y == 1-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #3]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #2
ADD X8, X8, #4
LDURSW X8, [x25, x8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [x26, #4]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #2
ADD X8, X8, #8
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #5]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 1-------------------

// -------------------Y == 2-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #3
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #6]
// sum += input[j + y][i + x] * weights[y][x]
// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #3
ADD X8, X8, #4
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #7]
// sum += input[j + y][i + x] * weights[y][x]
// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
LSL X8, X24, #3
ADD X8, X8, #8
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #8]
// sum += input[j + y][i + x] * weights[y][x]
// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------


// -------------------END Y == 2-------------------
LDURSB X8, [X27]
ADD X23, X23, X8
// x23 = relu(sum + *bias)

// relu sum for j = 0, i = 0

// x8 = &output + (j * (n - 2) + i) * 4
//relu_skip
ASR X9, X23, #64
EOR x9, x9, #-1
AND X23, X23, X9
STURW X23, [X28]
ADD X20, X20, #1
B convolution_loop_i


convolution_loop_j:
// exit if j >= n - 2
SUB X8, X24, #2

CMP X19, X8
// i = 0
MOV X20, XZR
// stall 5 here
B.GE convolution_exit_loop_j

// -------------------I == 0-------------------

// -------------------Y == 0-------------------
// x = 0

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X23, X8, X9
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL X8, X8, #2
ADD X8, X8, #4
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #1]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL X8, X8, #2
ADD X8, X8, #8
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #2]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 0-------------------

// -------------------Y == 1-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
ADD x8, x8, x24
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #3]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
ADD x8, x8, x24
ADD X8, X8, #1
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #4]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
ADD x8, x8, x24
ADD X8, X8, #2
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #5]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 1-------------------

// -------------------Y == 2-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL X8, X8, #2
LSL x9, x24, #3
ADD X8, X8, X9
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #6]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL x9, x24, #1
ADD X8, X8, X9
ADD X8, X8, #1
LSL X8, X8, #2
LDURSW X8, [X25, X8]

// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #7]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, x19, x24
LSL x9, x24, #1
ADD X8, X8, X9
ADD X8, X8, #2
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #8]
// sum += input[j + y][i + x] * weights[y][x]


// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------


// -------------------END Y == 2-------------------

LDURSB X8, [X27]

ADD x20, x20, #1
ADD X23, X23, X8
// x23 = relu(sum + *bias)

CMP X23, #0
// x8 = &output + (j * (n - 2) + i) * 4
// stall 5 here, neccessary
SUB X8, X24, #2
MUL X8, X8, X19
LSL X8, X8, #2
ASR X9, X23, #64
EOR x9, x9, #-1
AND X23, X23, X9
STURW X23, [X28, X8]
// --------------------END I == 0--------------------


convolution_loop_i:
// exit if i >= n - 2
SUB X8, X24, #2


CMP X20, X8
// stall 5 here
B.GE convolution_exit_loop_i

// -------------------Y == 0-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X20
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X23, X8, X9
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X20
ADD X8, X8, #1
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #1]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X20
ADD X8, X8, #2
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #2]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 0-------------------

// -------------------Y == 1-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X24
ADD X8, X8, X20
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #3]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X24
ADD X8, X8, X20
ADD X8, X8, #1
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #4]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
ADD X8, X8, X24
ADD X8, X8, X20
ADD X8, X8, #2
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #5]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------

// -------------------END Y == 1-------------------

// -------------------Y == 2-------------------

// -------------------X == 0-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
LSL X9, X24, #1
ADD X8, X8, X9
ADD X8, X8, X20
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #6]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 0-------------------

// ---------------------X == 1---------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
LSL X9, X24, #1
ADD X8, X8, X9
ADD X8, X8, X20
ADD X8, X8, #1
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #7]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 1-------------------

// ---------------X == 2-------------------
// x8 = input[j + y][i + x] = *(input + ((j + y) * n + i + x) * 4)
MUL X8, X19, X24
LSL X9, X24, #1
ADD X8, X8, X9
ADD X8, X8, X20
ADD X8, X8, #2
LSL X8, X8, #2
LDURSW X8, [X25, X8]
// x9 = weights[y][x] = *(weights + y * 3 + x)
LDURSB X9, [X26, #8]
// sum += input[j + y][i + x] * weights[y][x]

// stall 1 here, acceptable to alternative
MUL X8, X8, X9
ADD X23, X23, X8
//-----------------END X == 2-------------------


// -------------------END Y == 2-------------------

// sum += *bias
LDURSB X8, [X27]
ADD X23, X23, X8
// x23 = relu(sum + *bias)
// x8 = &output + (j * (n - 2) + i) * 4
//SUB X8, X24, #2
// stall 5 here, neccessary
SUB X8, X24, #2
MUL X8, X8, X19
ADD X8, X8, X20
LSL X8, X8, #2
ASR X9, X23, #64
EOR x9, x9, #-1
AND X23, X23, X9
STURW X23, [X28, X8]
// i++
ADD X20, X20, #1
// stall 5 here, neccessary
B convolution_loop_i


convolution_exit_loop_i:
// j++
ADD X19, X19, #1
// stall 5 here
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
//BR LR

exit:
// Exit sys call terminates program
MOV X8, #93
SVC 0