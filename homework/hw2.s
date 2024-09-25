_start:
mov x0, #0
mov x1, #10
main: 
    loop:
        sub x1, x1, #1
        ADD x0, x0, #2
        cmp x1 #0
        B.LE DONE
        B LOOP
DONE: