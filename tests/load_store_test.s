.data
test_byte: .byte -67
test_hword: .hword -282
test_word: .word -100000
test_dword: .dword -1000000000

.text
    .global _start

_start:
main:
    ldur x9, =test_byte
    ldur x10, =test_hword
    ldur x11, =test_word
    ldur x12, =test_dword

    ldurb x1, [x9, #0]
    ldursb x2, [x9, #0]
    ldurh x3, [x10, #0]
    ldursh x4, [x10, #0]
    ldursw x5, [x11, #0]

    mov x8, #-88
    sturb x8, [x9, #0]
    ldursb x6, [x9, #0]
    ldurb x7, [x9, #0]

    sturh x8, [x10, #0]
    ldursh x9, [x10, #0]
    ldurh x10, [x10, #0]
