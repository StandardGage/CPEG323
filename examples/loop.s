// Calculate the sum of numbers

.data
numbers:    .byte -67, -114, -96, -54, -120, -128

.text
.global _start
_start:
    // Load base address of numbers
    LDUR x0, =numbers
    // i = 0
    MOV X9, XZR
    // sum = 0
    MOV X10, XZR
    // numbers_size = 6
    MOV X11, #6
    loop:
        // if i >= 6 exit loop
        CMP X9, X11
        B.GE exit
        // x8 = numbers[i]
        LDURSB X8, [X0, X9]
        // sum += x8
        ADD X10, X10, X8
        // increment i
        ADD X9, X9, #1
        B loop
    exit:
    // Exit syscall
    MOV x8, #93
    SVC 0  


