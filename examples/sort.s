.data
	numbers: .dword 9, 8, 2, 3, 6

.text
    .global _start

_start:
main:
	ldur x0, =numbers
	mov x1, #5
	bl sort
	mov x8, #93
	svc 0

swap:
	lsl x10, x1, #3
	add x10, x0, x10
	ldur x9, [x10, #0]
	ldur x11, [x10, #8]
	stur x11, [x10, #0]
	stur x9, [x10, #8]
	br lr

sort:
	sub sp, sp, #32
	stur lr, [sp, #24]
	stur x21, [sp, #16]
	stur x20, [sp, #8]
	stur x19, [sp, #0]
	mov x19, xzr

for1tst:
	cmp x19, x1
	b.ge exit1
	mov x22, x1
	sub x20, x19, #1

for2tst:
	cmp x20, xzr
	b.lt exit2
	lsl x10, x20, #3
	add x11, x0, x10
	ldur x12, [x11, #0]
	ldur x13, [x11, #8]
	cmp x12, x13
	b.le exit2
	mov x1, x20
	bl swap
	sub x20, x20, #1
	b for2tst

exit2:
	mov x1, x22
	add x19, x19, #1
	b for1tst

exit1:
	ldur x19, [sp, #0]
	ldur x20, [sp, #8]
	ldur x21, [sp, #16]
	ldur lr, [sp, #24]
	add sp, sp, #32
	br lr


