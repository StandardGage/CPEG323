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
	ldur x11,[x10, #8]
	stur x11,[x10, #0]
	stur x9, [x10, #8]
	br lr

sort:
	sub sp, sp, #32
	stur lr, [sp, #24]
	stur x21, [sp, #16]
	stur x20, [sp, #8]
	stur x19, [sp, #0]
	mov x14, xzr

for1tst:
	cmp x14, x1
	b.ge exit1
	sub x15, x14, #1

for2tst:
	cmp x15, xzr
	b.lt exit2
	lsl x10, x15, #3
	add x11, x0, x10
	ldur x12, [x11, #0]
	ldur x13, [x11, #8]
	cmp x12, x13
	b.le exit2
	stur x12, [x11, #8]
	stur x13, [x11, #0]
	sub x15, x15, #1
	b for2tst

exit2:
	add x14, x14, #1
	b for1tst
exit1:
	br lr
