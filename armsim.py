import re
import sys
import os

'''
*******************
* ArmSim Overview *
*******************
The goal of this program is to simulate an arm64 processor
executing a compiled .s file. It attempts to be compatible
with the format of gnu assembler files and supports a subset
of the instructions and directives. The basic operation of
the simulator is that it first reads in a .s file line by 
line and separates the input into code and symbol declarations. 
The data in static memory is simulated with a python list, where
each element represents one byte as an int. It attempts to execute 
each line of code by matching against regular expressions that encode 
the instruction format, and updating global variables appropriately 
based on that execution. All text is converted to lower case, 
meaning that identifiers are not case sensitive 
(so variable = VARIABLE).
Currently supported:
  System Calls:
    read      0x3f  (63) --stdin only
    write     0x40  (64) --stdout only
    getrandom 0x116 (278)
  Labels:
    Can be any text (current no numbers) prepended with
    any number of periods or underscores and should end in 
    a colon. The same label cannot be declared twice. Since 
    text is converted to lowercase, LABEL: and label: would 
    count as the same. Labels must be declared on their OWN 
    line.
  Directives:
    .data    (declare a region of initialized data)
        .asciz   (declare a string in the .data section)
        .dword   (declare an array of [8 bytes] dwords in the .data section)
        .word    (declare an array of [4 bytes] words in the .data section)
        .hword   (declare an array of [2 bytes] half words in the .data section)
        .byte    (declare an array of bytes in the .data section)
        =        (assignment of a variable to a constant value within the .data section)
        = . -      (find the length of the previously declared item within the .data section)
    .bss     (declare a region of unitialized data)
        .space   (declare an empty buffer in the .bss section)

  Instructions:
    **{s} means that 's' can be optionally added to the end of an
    instruction to make the result affect the flags**
    rd      = destination register
    *rt/rt2 = target register*
    rn      = first register operand
    rm      = second register operand
    imm     = immediate value (aka a number)
    ldursw  rt, [rn]
    ldursw  rt, [rn, imm]
    ldursw  rt, [rn, rm]
    ldurh   rt, [rn]
    ldurh   rt, [rn, imm]
    ldursh   rt, [rn]
    ldursh   rt, [rn, imm]
    ldurb   rt, [rn]
    ldurb   rt, [rn, imm]
    ldursb  rt, [rn]
    ldursb  rt, [rn, imm]
    ldur    rd, =<var>
    ldur    rt, [rn]
    ldur    rt, [rn, imm]
    ldur    rt, [rn, rm]
    sturh   rt, [rn]
    sturh   rt, [rn, imm]
    sturh   rt, [rn, rm]
    sturb   rt, [rn]
    sturb   rt, [rn, imm]
    sturb   rt, [rn, rm]
    stur    rt, [rn]
    stur    rt, [rn, imm]
    stur    rt, [rn, rm]
    mov     rd, imm
    mov     rd, rn
    sub{s}  rd, rn, imm
    sub{s}  rd, rn, rm
    add{s}  rd, rn, imm
    add{s}  rd, rn, rm
    asr     rd, rn, imm
    lsl     rd, rn, imm
    udiv    rd, rn, rm
    sdiv    rd, rn, rm
    mul     rd, rn, rm
    and{s}  rd, rn, imm
    and{s}  rd, rn, rm
    orr{s}  rd, rn, imm
    orr{s}  rd, rn, rm
    eor{s}  rd, rn, imm
    cmp     rn, rm
    cbnz    rn, <label>
    cbz     rn, <label>
    b       <label>
    b.gt    <label>
    b.ge    <label>
    b.lt    <label>
    b.le    <label>
    b.eq    <label>
    b.ne    <label>
    b.mi    <label>
    b.pl    <label>
    bl      <label>
    br lr
    svc 0   


Comments (Must NOT be on same line as stuff you want read into the program):
  //text
  /*text*/
  /*
  text
  */
'''

'''
Global state variables
'''
# list to hold the instructions
asm = []
STACK_SIZE = 4096
# heap will be 4 pages
HEAP_SIZE = 0x4000
# points to the original break, which comes after the static data
original_break = 0
# points to current break
brk = 0
# dict of register names to values. Will always be numeric values
reg = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 0, 'x7': 0, 'x8': 0, 'x9': 0, 'x10': 0,
       'x11': 0, 'x12': 0, 'x13': 0, 'x14': 0, 'x15': 0, 'x16': 0, 'x17': 0, 'x18': 0, 'x19': 0, 'x20': 0,
       'x21': 0, 'x22': 0, 'x23': 0, 'x24': 0, 'x25': 0, 'x26': 0, 'x27': 0, 'x28': 0, 'fp': 0, 'lr': 0, 'sp': 0,
       'xzr': 0}
# program counter
pc = 0
# Note: Python doesn't really have overflow and it would
# be a pain to simulate, so the v (signed overflow) flag
# is implicitly zero
# negative flag
n_flag = False
# zero flag
z_flag = False

'''
dict to hold how often a label has been seen. Intialized in the
run() procedure, then updated in the main loop every time a label is
hit. Since the BL instruction does not cause the pc to actually
land on the label, label_hit_counts must also be updated in execute()
when a BL instruction is matched (colon must be included)
'''
label_hit_counts = {}

'''
dict to hold "external" labels that can be targets for BL.
The key is a label (including colon) and the value is a python
function
'''
linked_labels = {}

'''
regexes for parsing instructions
'''
register_regex = '(?:lr|fp|sp|xzr|(?<!\w)x[1-2]\d(?!\w)|(?<!\w)x\d(?!\w))'
num_regex = '[-]?(?:0x[0-9a-f]+|\d+)'
var_regex = '[a-z_]+\w*'
label_regex = '[.]*\w+'
'''
regex explanations:
------------------
register_regex:
    (?:
        we will match any of the options between the |
    (?<!w)
        negative lookbehind is so that we don't match hex numbers like 0x40
        as registers or labels that happen to have register names
    x[1-2]\d
        matches registers x10 - x29 (fp is used instead of x29, should fix this)
    (?!\w)
        negative lookahead to ensure that cases like x222 aren't matched
    (?<!0)x\d(?!\w)
        same explanations as above, but this is for registers x0 - x9
        again, we don't want to match registers like x90, so the negative
        lookahead is used
num_regex:
    [-]?(?:0x[0-9a-f]+|\d+)
    [-]?
        optionally matches a negative sign at the beginning
    (?:0x[0-9a-f]+|[0-9]+)
        matches either a hex number starting with 0x or a regular number
label_regex    
    [.]*
        a label can start with zero or more periods
    \w+
        followed by one or more alpanumeric symbols or underscore
'''

'''
A map of string to int, where int will either be
an index into the mem array or a size in bytes.
Basically, vars declared with a : will be addresses and
vars declared with = will be literals
Additionally the directive type will be stored in the following way:
-an key in the form <var>_TYPE_ will map to
0 -> asciz
1 -> dword
2 -> space
3 -> hword
4 -> word
5 -> byte
NB. vars declared with = (ie length variables) are just stored in
sym_table as numbers, so they don't have a type
Types are stored primarily for the get_data procedure
'''
sym_table = {}

'''
Data is stored as a list. Each element is an int that represents a byte. 
String data gets "converted" by doing list(bytes(str,'ascii')) and numbers 
get converted into a list from their byte representation using list(int.tobytes()).
It is accessed with an index and a size using the format [addr:addr+size].
The stack pointer also points to the end of this list and grows down.
It's first filled with the stack, then static data, then the heap. This
ensures that increasing the heap does not shift the stack or static data.
The sp (stack pointer) register will point to the end of its section
and the heap pointer will point beginning of it's section to start
Thus, we get the following diagram

| stack | static | heap |
        ^        ^     
    <--sp        hp -->
'''
mem = []

'''
Static Rule Variables:
Specify properties that a program must have
--disallow certain instructions
--require/forbid recursion
'''
# A set that contains the mnemonic of instructions that you don't want used
# for a particular run of the the program
forbidden_instructions = set()

# recursion flags
forbid_recursion = False
require_recursion = False
# loop flag
forbid_loops = False

# dead code flag
check_dead_code = False

# set to add labels that should be recursively called
# (do not include colon)
recursive_labels = set()

# Performance Flags
cycle_count = 0
execute_count = 0
ld_cycle, ld_dst = -1, -1
flag_cycle = -1
last_dst = -1

'''
This procedure reads the lines of a program (which can be a .s file
or just a list of assembly instructions) and populates the
sym_table, mem, and asm data structures. It uses
boolean flags to determine which datastructure is currently
being populated. These flags change upon encountering specific
keywords. Those keywords are .data or .bss for declaring constants
and buffers and main: or _start: for code. 
'''


def parse(lines) -> None:
    global STACK_SIZE, HEAP_SIZE, heap_pointer, original_break, brk
    # booleans for parsing .s file
    comment = False
    code = False
    data = False
    bss = False

    # allocate the stack and set the stack pointer
    mem.extend(list([0] * STACK_SIZE))
    reg['sp'] = len(mem) - 1
    '''
    This is a counter that is used to assign an "address" in mem
    to a symbol. Basically the value in sym_table when a key is one of 
    the user defined variables. It's incremented for every variable encountered
    by the size of the data stored in mem
    '''
    index = len(mem)

    for line in lines:
        line = line.strip()
        # convert multiple spaces into one space
        line = re.sub('[ \t]+', ' ', line)
        if ('/*' in line and '*/' in line): continue
        if ('//' in line): continue
        if ("/*" in line): comment = True;continue
        if ("*/" in line): comment = False;continue
        if (".data" in line): data = True;code = False;bss = False;continue
        if (".bss" in line): data = False;code = False;bss = True;continue
        if ("main:" in line or "_start:" in line): code = True;data = False;bss = False;continue
        if (code and not comment and len(line) > 0): line = line.lower();asm.append(line)
        if ((data or bss) and not comment):
            # remove quotes and whitespace surrouding punctuation
            # spaces following colons and periods are not touched so
            # that string literals are not altered
            line = re.sub('[ ]*:', ':', line)
            line = re.sub('[ ]*\.', '.', line)
            line = re.sub('[ ]*-[ ]*', '-', line)
            line = re.sub('[ ]*=[ ]*', '=', line)
            '''
            When encountering something like s: .asciz "a"
            we want to make s a new key in the sym_table dict and 
            set its value equal to the second element after
            splitting on the string ":.asciz". Additionally
            we save the length of the string in a "shadow entry"
            in sym_table in case someone wants to find the length
            using the -. idiom. The string gets converted to bytes
            before it is written to mem
            '''
            if (re.match('.*:\.asciz.*', line)):
                # Don't convert string literals to lower case, so split on quote
                # and everything to the left becomes lower
                line = line[0:line.find('\"')].lower() + line[line.find('\"'):]
                # remove quote characters
                line = re.sub('["]', '', line)
                # escape characters get mangled to \\<char>, convert to \<char>
                # for now just tab, carriage return, and newline
                line = line.replace('\\n', '\n')
                line = line.replace('\\t', '\t')
                line = line.replace('\\r', '\r')
                line = line.split(":.asciz ")
                sym_table[line[0]] = index
                sym_table[line[0] + "_SIZE_"] = len(line[1])
                sym_table[line[0] + "_TYPE_"] = 0
                mem.extend(list(bytes(line[1], 'ascii')))
                index += len(list(line[1]))
                continue
            '''
            A similar procedure is done the .space directive is used
            We first check if a previously declared variable is being
            used to determine the size. If so we fetch it and use that,
            otherwise we just use the number provided. We append a list
            with n zero values to mem where n is the size we found
            Additionally, the size is stored in a shadow entry
            '''
            if (re.match('.*:\.space.*', line)):
                line = line.lower()
                line = line.split(":.space ")
                size = sym_table[line[1]] if line[1] in sym_table else int(line[1])
                mem.extend(list([0] * size))
                sym_table[line[0]] = index
                sym_table[line[0] + "_TYPE_"] = 2
                sym_table[line[0] + "_SIZE_"] = size
                index += size
                continue

            '''
            The .dword directive is followed by a comma separated list
            of numbers. Each number will be an 8 byte entry in mem.
            Additionally, the _SIZE_ shadow entry will be created
            '''
            if (re.match('.*:\.dword.*', line)):
                line = line.lower()
                line = line.split(":.dword")
                numbers = list(map(int, line[1].split(',')))
                # each number is 8 bytes
                size = len(numbers) * 8
                for n in numbers:
                    signed = not (0 <= n <= 255 ** 8)
                    mem.extend(list(int.to_bytes(n, 8, 'little', signed=signed)))

                sym_table[line[0]] = index
                sym_table[line[0] + "_SIZE_"] = size
                sym_table[line[0] + "_TYPE_"] = 1
                index += size
                continue

            '''
            The .word directive is followed by a comma separated list
            of numbers. Each number will be a 4 byte entry in mem.
            Additionally, the _SIZE_ shadow entry will be created
            '''
            if (re.match('.*:\.word.*', line)):
                line = line.lower()
                line = line.split(":.word")
                numbers = list(map(int, line[1].split(',')))
                # each number is 4 bytes
                size = len(numbers) * 4
                for n in numbers:
                    signed = not (0 <= n <= 255 ** 4)
                    mem.extend(list(int.to_bytes(n, 4, 'little', signed=signed)))

                sym_table[line[0]] = index
                sym_table[line[0] + "_SIZE_"] = size
                sym_table[line[0] + "_TYPE_"] = 3
                index += size
                continue

            '''
            The .hword directive is followed by a comma separated list
            of numbers. Each number will be a 2 byte entry in mem.
            Additionally, the _SIZE_ shadow entry will be created
            '''
            if (re.match('.*:\.hword.*', line)):
                line = line.lower()
                line = line.split(":.hword")
                numbers = list(map(int, line[1].split(',')))
                # each number is 2 bytes
                size = len(numbers) * 2
                for n in numbers:
                    signed = not (0 <= n <= 255 ** 2)
                    mem.extend(list(int.to_bytes(n, 2, 'little', signed=signed)))

                sym_table[line[0]] = index
                sym_table[line[0] + "_SIZE_"] = size
                sym_table[line[0] + "_TYPE_"] = 4
                index += size
                continue

            '''
            The .byte directive is followed by a comma separated list
            of numbers. Each number will be a 1 byte entry in mem.
            Additionally, the _SIZE_ shadow entry will be created
            '''
            if (re.match('.*:\.byte.*', line)):
                line = line.lower()
                line = line.split(":.byte")
                numbers = list(map(int, line[1].split(',')))
                # each number is 1 bytes
                size = len(numbers)
                for n in numbers:
                    signed = not (0 <= n <= 255)
                    mem.extend(list(int.to_bytes(n, 1, 'little', signed=signed)))

                sym_table[line[0]] = index
                sym_table[line[0] + "_SIZE_"] = size
                sym_table[line[0] + "_TYPE_"] = 5
                index += size
                continue

            '''
            If using the len=.-str idiom to store str length, we
            lookup the length of str that we stored in sym_table
            dict when handling .asciz in the format str_SIZE_ 
            '''
            if (re.match('(.)+=.-(.)+', line)):
                line = line.lower()
                line = line.split("=.-")
                if (line[1] not in sym_table):
                    raise KeyError("Can't find length of undeclared variable " + line[1])
                sym_table[line[0]] = sym_table[line[1] + "_SIZE_"]
                continue
            '''
            This is for when constants are declared with the = sign
            If assigning an existing value, look it up in the sym_table
            and if it's not there, then assume a number is being assigned. 
            '''
            if (re.match('(.)+=[a-z0-9]+', line)):
                line = line.lower()
                line = line.split("=")
                value = 0
                if (line[1] in sym_table):
                    sym_table[line[0]] = sym_table[line[1]]
                else:
                    sym_table[line[0]] = int(line[1])

    # set the break variables to the end of static memory
    original_break = index
    brk = original_break
    assert brk == len(mem), \
        "mem list likely incorrect- brk: {} len(mem):{}".format(brk, len(mem))
    # extend mem to make room for the stack, then set the stack pointer
    # mem.extend(list([0]*HEAP_SIZE))


'''
This procedure dispatches and executes the provided line
of assembly code. In order to deal with the myriad
addressing modes, a regex method is used to match
the line to the appropriate action. Once an instruction is matched,
the arguments are extracted with regular epressions.The procedure 
returns after executing the matched instruction. If no match is
found an exception is thrown. Both hexadecimal
and decimal immediate values are supported. The register
naming convention is rd for destination register, rn
for the first arg register and rm for the second arg regsiter
Notes:
-int(str,0) means that both numerical strings and hex strings
will be properly converted
-Error message is very general, so any syntax errors or use 
of unsupported instructions will throw the same error.
-If an illegal register is used, it will trigger a syntax error
-If a register is used in a branch instr that doesn't take them,
an error is raised
'''


def execute(line: str):
    global pc, n_flag, z_flag, label_hit_counts, mem
    global original_break, brk, STACK_SIZE, HEAP_SIZE
    global register_regex, num_regex, var_regex, label_regex
    global cycle_count, execute_count, ld_cycle, ld_dst
    global flag_cycle, last_dst
    current_cycle = cycle_count
    cycle_count += 1
    execute_count += 1
    last_dst = None

    # remove spaces around commas
    line = re.sub('[ ]*,[ ]*', ',', line)
    # octothorpe is optional, remove it
    line = re.sub('#', '', line)

    # use abbreviations for the regexes
    rg = register_regex
    num = num_regex
    var = var_regex
    lab = label_regex

    # all labels in program (better feedback for typos/malformed branches
    # [:-1] is so that the colon in the label is not included
    labels = [l[:-1] for l in asm if (re.match('{}:'.format(lab), l))]

    '''ldursw instructions'''
    # ldursw rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('ldursw {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 4):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 4 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 4]), 'little', signed=True)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldursw rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldursw {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 4 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 4]), 'little', signed=True)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldursw rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldursw {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle) <= 2:
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {} at {}".format(line, addr))
        # load 4 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 4]), 'little', signed=True)
        ld_cycle = current_cycle
        ld_dst = rt
        return

    '''ldurh instructions'''
    # ldurh rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('ldurs?h {},\[{}\]$'.format(rg, rg), line)):
        signed = 'ldursh' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 2 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 2]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldurh rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldurs?h {},\[{},{}\]$'.format(rg, rg, num), line)):
        signed = 'ldursh' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 2 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 2]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldurh rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldurs?h {},\[{},{}\]$'.format(rg, rg, rg), line)):
        signed = 'ldursh' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle) <= 2:
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 2 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 2]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return

    '''ldurb instructions'''
    # ldurb rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('ldurs?b {},\[{}\]$'.format(rg, rg), line)):
        signed = 'ldursb' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 1 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 1]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldurb rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldurs?b {},\[{},{}\]$'.format(rg, rg, num), line)):
        signed = 'ldursb' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 1 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 1]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldurb rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldurs?b {},\[{},{}\]$'.format(rg, rg, rg), line)):
        signed = 'ldursb' in line
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle) <= 2:
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 1 byte starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 1]), 'little', signed=signed)
        ld_cycle = current_cycle
        ld_dst = rt
        return

    '''
    ldur instructions
    '''
    # ldur rt, =<var>
    if (re.match('ldur {},={}$'.format(rg, var), line)):
        rt = re.findall(rg, line)[0]
        v = re.findall('=' + var, line)[0][1:]
        reg[rt] = sym_table[v]
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldur rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('ldur {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 8 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 8]), 'little')
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldur rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldur {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 8 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 8]), 'little')
        ld_cycle = current_cycle
        ld_dst = rt
        return
    # ldur rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('ldur {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        # load 8 bytes starting at addr and convert to int
        reg[rt] = int.from_bytes(bytes(mem[addr:addr + 8]), 'little')
        ld_cycle = current_cycle
        ld_dst = rt
        return
    '''sturw instruction'''
    # sturw rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('sturw {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))

        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 4] = register_bytes[:4]
        return
    # sturw rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturw {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 4] = register_bytes[:4]
        return
    # sturw rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturw {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 4] = register_bytes[:4]
        return
    '''sturh instruction'''
    # sturh rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('sturh {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))

        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 2] = register_bytes[:2]
        return
    # sturh rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturh {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 2] = register_bytes[:2]
        return
    # sturh rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturh {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 2):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 2] = register_bytes[:2]
        return
    '''sturb instruction'''
    # sturb rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('sturb {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 1] = register_bytes[:1]
        return
    # sturb rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturb {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 1] = register_bytes[:1]
        return
    # sturb rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('sturb {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 1):
            raise ValueError("out of bounds memory access: {}".format(line))
        register_bytes = list(int.to_bytes((reg[rt]), 8, 'little', signed=True))
        mem[addr:addr + 1] = register_bytes[:1]
        return
    '''
    stur instructions
    '''
    # stur rt, [rn]
    # dollar sign so it doesn't match post index
    if (re.match('stur {},\[{}\]$'.format(rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        addr = reg[rn]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        mem[addr:addr + 8] = list(int.to_bytes((reg[rt]), 8, 'little'))
        return
    # stur rt, [rn, imm]
    # dollar sign so it doesn't match pre index
    if (re.match('stur {},\[{},{}\]$'.format(rg, rg, num), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        imm = int(re.findall(num, line)[-1], 0)
        addr = reg[rn] + imm
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        mem[addr:addr + 8] = list(int.to_bytes((reg[rt]), 8, 'little'))
        return
    # stur rt, [rn, rm]
    # dollar sign so it doesn't match pre index
    if (re.match('stur {},\[{},{}\]$'.format(rg, rg, rg), line)):
        rt = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        addr = reg[rn] + reg[rm]
        # check for out of bounds mem access
        if (addr < reg['sp'] or addr > len(mem) - 8):
            raise ValueError("out of bounds memory access: {}".format(line))
        mem[addr:addr + 8] = list(int.to_bytes((reg[rt]), 8, 'little'))
        return
    '''
    mov instructions
    '''
    # mov rd, imm
    if (re.match('mov {},{}$'.format(rg, num), line)):
        rd = re.findall(rg, line)[0]
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = imm
        last_dst = rd
        return
    # mov rd, rn
    if (re.match('mov {},{}$'.format(rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += 1
        reg[rd] = reg[rn]
        last_dst = rd
        return
    '''
    arithmetic instructions
    '''
    # asr rd, rn, imm
    if (re.match('asr {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle) <= 2:
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] >> imm
        last_dst = rd
        return
    # asr rd, rn, rm
    if (re.match('asr {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] >> reg[rm]
        last_dst = rd
        return
    # lsr rd, rn, imm
    if (re.match('lsr {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = (reg[rn] & 0xFFFFFFFFFFFFFFFF) >> imm
        last_dst = rd
        return
    # lsr rd, rn, rm
    if (re.match('lsr {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = (reg[rn] & 0xFFFFFFFFFFFFFFFF) >> reg[rm]
        last_dst = rd
        return
    # lsl rd, rn, imm
    if (re.match('lsl {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = (reg[rn] << imm) & 0xFFFFFFFFFFFFFFFF
        last_dst = rd
        return
    # lsl rd, rn, rm
    if (re.match('lsl {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = (reg[rn] << reg[rm]) & 0xFFFFFFFFFFFFFFFF
        last_dst = rd
        return
    # add{s} rd, rn, imm
    if (re.match('adds? {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 1):
            cycle_count += 1
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] + imm
        if ('adds' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # add{s} rd, rn, rm
    if (re.match('adds? {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] + reg[rm]
        if ('adds' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # sub{s} rd, rn, imm
    if (re.match('subs? {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] - imm
        if ('subs' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # sub{s} rd, rn, rm
    if (re.match('subs? {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] - reg[rm]
        if ('subs' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # mul rd, rn, rm
    if (re.match('mul {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] * reg[rm]
        last_dst = rd
        cycle_count += 4
        return
    # For now treat un/signed division the same, since everything
    # is signed in python, but separate in case this changes
    # udiv rd, rn, rm
    if (re.match('udiv {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        # IMPORTANT: use integer division, not floating point
        reg[rd] = reg[rn] // reg[rm]
        last_dst = rd
        return
    # sdiv rd, rn, rm
    if (re.match('sdiv {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        # IMPORTANT: use integer division, not floating point
        reg[rd] = reg[rn] // reg[rm]
        last_dst = rd
        return
    '''
    compare instructions
    '''
    # cmp rn, rm
    if (re.match('cmp {},{}$'.format(rg, rg), line)):
        rn = re.findall(rg, line)[0]
        rm = re.findall(rg, line)[1]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        assert rm != 'sp', "2nd register in cmp can't be sp"
        z_flag = True if reg[rn] == reg[rm] else False
        n_flag = True if reg[rn] < reg[rm] else False
        flag_cycle = current_cycle
        last_dst = rn
        return
    # cmp rn, imm
    if (re.match('cmp {},{}$'.format(rg, num), line)):
        rn = re.findall(rg, line)[0]
        if (ld_dst == rn) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        imm = int(re.findall(num, line)[-1], 0)
        z_flag = True if reg[rn] == imm else False
        n_flag = True if reg[rn] < imm else False
        flag_cycle = current_cycle
        last_dst = rn
        return
    '''
    logical instructions
    '''
    # and{s} rd, rn, imm
    if (re.match('ands? {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] & imm
        if ('ands' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # and{s} rd, rn, rm
    if (re.match('ands? {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] & reg[rm]
        if ('ands' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # orr{s} rd, rn, imm
    if (re.match('orrs? {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] | imm
        if ('orrs' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # orr{s} rd, rn, rm
    if (re.match('orrs? {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] | reg[rm]
        if ('orrs' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # eor{s} rd, rn, imm
    if (re.match('eors? {},{},{}$'.format(rg, rg, num), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        if ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += (current_cycle - ld_cycle)
        imm = int(re.findall(num, line)[-1], 0)
        reg[rd] = reg[rn] ^ imm
        if ('eors' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    # eor{s} rd, rn, rm
    if (re.match('eors? {},{},{}$'.format(rg, rg, rg), line)):
        rd = re.findall(rg, line)[0]
        rn = re.findall(rg, line)[1]
        rm = re.findall(rg, line)[2]
        if (ld_dst == rn or ld_dst == rm) and (current_cycle - ld_cycle <= 2):
            cycle_count += 1
        reg[rd] = reg[rn] ^ reg[rm]
        if ('eors' in line):
            n_flag = True if (reg[rd] < 0) else False
            z_flag = True if (reg[rd] == 0) else False
            flag_cycle = current_cycle
        last_dst = rd
        return
    '''
    branch instructions
    NB. A value error is raised if a register is included where it shouldn't be
    '''
    # cbnz rn,<label>
    if (re.match('cbnz {},{}$'.format(rg, lab), line)):
        if (len(re.findall(rg, line)) != 1): raise ValueError("cbnz takes one register")
        rn = re.findall(rg, line)[0]
        if last_dst == rn:
            cycle_count += 1
        elif ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += 3 - (current_cycle - ld_cycle)
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (reg[rn] != 0):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # cbz rn, <label>
    if (re.match('cbz {},{}$'.format(rg, lab), line)):
        if (len(re.findall(rg, line)) != 1): raise ValueError("cbz takes one register")
        rn = re.findall(rg, line)[0]
        if last_dst == rn:
            cycle_count += 1
        elif ld_dst == rn and (current_cycle - ld_cycle <= 2):
            cycle_count += 3 - (current_cycle - ld_cycle)
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (reg[rn] == 0):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b <label>
    if (re.match('b {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("b takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        pc = asm.index(label + ':')
        cycle_count += 1
        return
    # b.lt <label>
    if (re.match('b\.?lt {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("blt takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (n_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.le <label>
    if (re.match('b\.?le {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("ble takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (n_flag or z_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.gt <label>
    if (re.match('b\.?gt {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bgt takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (not z_flag and not n_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.ge <label>
    if (re.match('b\.?ge {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bge takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (not n_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.eq <label>
    if (re.match('b\.?eq {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("beq takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (z_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.ne <label>
    if (re.match('b\.?ne {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bne takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (not z_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.mi <label>
    if (re.match('b\.?mi {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bmi takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (n_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # b.pl <label>
    if (re.match('b\.?pl {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bpl takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1]
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        if (not n_flag or z_flag):
            pc = asm.index(label + ':')
            cycle_count += 1
        return
    # bl <label>
    # bl can branch to a local assembly procedure or to an externally defined
    # python function
    if (re.match('bl {}$'.format(lab), line)):
        if (len(re.findall(rg, line)) != 0): raise ValueError("bl takes no registers")
        # last match is the label
        label = re.findall(lab, line)[-1] + ':'
        if (current_cycle - flag_cycle <= 1):
            cycle_count += 1
        reg['lr'] = pc
        # label_hit_counts must be updated here to count procedure calls
        if (label in label_hit_counts.keys()):
            label_hit_counts[label] += 1
        # behavior depends if local or external label
        if (label in linked_labels):
            linked_labels[label]()
        else:
            pc = asm.index(label)
        cycle_count += 1
        return
    # br lr
    if (re.match('br lr$', line)):
        addr = reg['lr']
        if (addr not in range(0, len(asm))):
            raise ValueError("ret: address in LR ({}) out of range".format(addr))
        pc = addr
        cycle_count += 1
        return
    '''
    system call handler
    Currently supported: Read and write to stdin/stdout, getrandom
    '''
    # svc 0
    if (re.match('svc 0$', line)):
        syscall = int(reg['x8'])
        # simulate exit by causing main loop to exit
        if (syscall == 93):
            pc = len(asm)
        # write
        elif (syscall == 64):
            assert reg['x0'] == 1, "Can only write to stdout! (x0 must contain #1)"
            length = reg['x2']
            addr = reg['x1']
            output = bytes(mem[addr:addr + length]).decode('ascii')
            # if the user wants to print a newline they have to include
            # it in their string
            print(output, end='')
        # read
        elif (syscall == 63):
            length = reg['x2']
            addr = reg['x1']
            enter = input()
            enter += '\n'
            # truncate input based on # of chars read
            enter = enter[:length]
            # store as bytes, not string
            mem[addr:addr + len(enter)] = list(bytes(enter, 'ascii'))
            # return value is # of bytes read
            reg['x0'] = len(enter)
        # brk
        elif (syscall == 214):
            new_brk = reg['x0']
            # invalid new_brk, return current brk
            if (new_brk < original_break):
                reg['x0'] = brk
            # original brk, reset heap_pointer (works with empty data section)
            elif (new_brk == original_break):
                brk = new_brk
                reg['x0'] = brk
                mem = mem[:original_break]
            # adjust brk
            else:
                # round up to the nearest page boundary of 4K bytes
                break_size = new_brk - original_break
                assert break_size >= 0, "System error: break_size should never be negative"
                page = (break_size + 0x1000) - break_size % 0x1000
                if (page > HEAP_SIZE): raise ValueError("break size of {} too large".format(break_size))
                # shink the heap
                if (len(mem) > page + original_break):
                    mem = mem[:page + original_break]
                # grow the heap
                else:
                    mem.extend([0] * page)
                # x0 has valid address, set brk to it
                brk = reg['x0']
        # getrandom
        elif (syscall == 278):
            addr = reg['x0']
            quantity = reg['x1']
            # the number of random bytes requested is written to mem
            mem[addr:addr + quantity] = list(os.urandom(quantity))
            reg['x0'] = quantity
        else:
            raise ValueError("Unsupported system call: {} ".format(syscall))
        return
    raise ValueError("Unsupported instruction or syntax error: " + line)


'''
Takes a variable declared in the data or bss section
and returns the data (always as a list)at that address in a format 
that makes sense according to the directive type. The directive type
was stored in sym_table during the parse stage:
0 -> asciz
1 -> dword
2 -> space
3 -> hword
4 -> word
5 -> byte
Since the size of each variable is stored we can print out all data

Examples:

Given
message: .asciz "hello world\n"
get_data('message')
returns the list
['h','e','l','l','o',' ','w','o','r','l','d','\n']

Given
array: .dword 89,80,83,88,86,82,87,81,84,85
get_data('array')
returns the list
[89,80,83,88,86,82,87,81,84,85]

Given
len = 12
get_data('len')
returns the list
[12]

Given
steps: .space 8
get_data('steps')
returns the list
[0,0,0,0,0,0,0]
(assuming nothing has been put there)
'''


def getdata(variable: str):
    if (variable + '_TYPE_' in sym_table):
        index = sym_table[variable]
        size = sym_table[variable + "_SIZE_"]
        # asciz
        if (sym_table[variable + '_TYPE_'] == 0):
            return list(bytes(mem[index:index + size]).decode('ascii'))
        # dword
        elif (sym_table[variable + '_TYPE_'] == 1):
            lst = []
            for i in range(0, size, 8):
                lst.append(int.from_bytes(bytes(mem[index + i:index + i + 8]), 'little'))
            return lst
        # space
        elif (sym_table[variable + '_TYPE_'] == 2):
            return list(bytes(mem[index:index + size]))
        # word
        elif (sym_table[variable + '_TYPE_'] == 3):
            lst = []
            for i in range(0, size, 4):
                lst.append(int.from_bytes(bytes(mem[index + i:index + i + 4]), 'little'))
            return lst
        # hword
        elif (sym_table[variable + '_TYPE_'] == 4):
            lst = []
            for i in range(0, size, 2):
                lst.append(int.from_bytes(bytes(mem[index + i:index + i + 2]), 'little'))
            return lst
        # byte
        elif (sym_table[variable + '_TYPE_'] == 5):
            lst = []
            for i in range(0, size, 1):
                lst.append(int.from_bytes(bytes(mem[index + i:index + i + 1]), 'little'))
            return lst
        else:
            print(variable + ': variable not found')
    else:
        return [sym_table[variable]]


'''
Procedure to check that predefined rules about the code 
have been adhered to
Currently checks:
--Code has been detected and parsed into the asm list
--The same label is not declared twice
--forbidden instructions are not used
--branches are calling existing labels
--looping is not used, depending on flag
--the only text that immediately follow an unconditional branch
(ret or b) is a label, or it should be the last instruction
Called in run() and debug(), so should be no
need to call this separately 
'''


def check_static_rules():
    global forbid_recursion, require_recursion, check_dead_code, label_regex
    # label regex
    lab = label_regex

    # Make sure code has been detected
    if (not asm):
        raise ValueError("no code detected (remember to include a _start: or main: label)")
    # check for disallowed instructions:
    # --extract mnemonics (string before the first space)
    mnemonics = [i.split(" ")[0] for i in asm if " " in i]
    forbid = set(mnemonics).intersection(forbidden_instructions)
    if (forbid): raise ValueError("Use of {} disallowed".format(forbid))

    # verify that labels have not be redeclared
    labels = [l for l in asm if (re.match('{}:'.format(lab), l))]
    if (len(labels) > len(set(labels))):
        raise ValueError("You can't declare the same label more than once")

    # check that all branch instructions call existing labels
    for instr in asm:
        if (re.match('br lr$', instr)):
            continue
        if (re.match('c?b(.*?)', instr)):
            label = re.findall(lab, instr)[-1] + ":"
            if (label not in asm and label not in linked_labels):
                raise ValueError(instr + " is calling a nonexistent label")

    # To check for looping:
    # --match any branch instruction except bl
    # --if its label occurs earlier in the instruction
    # listing than the branch, it is a loop
    looped = False
    if (forbid_loops):
        for i in range(0, len(asm) - 1):
            # match branches except for bl
            if (re.match('c?b(?!l )', asm[i])):
                # last match is the label
                label = re.findall(lab, asm[i])[-1]
                if (asm.index(label + ':') < i):
                    looped = True
        if (looped):
            raise ValueError("you cannot loop")

    if (check_dead_code):
        # Check for dead code after ret or b instruction
        # The only instr that should come after a ret or b is a label
        for i in range(0, len(asm) - 1):
            # don't care about last instruction
            if (i != len(asm) - 1):
                if (asm[i] == 'br lr' or re.match('b {}'.format(lab), asm[i])):
                    assert re.match(lab + ':', asm[i + 1]), \
                        "Dead code detected after instruction {} " + asm[i]


'''
This procedure runs the code normally to the end. Exceptions are raised
for violated static checks, stack overflow, and if recursion is (un)used
contrary to the forbid/require recursion flags. The program is considered to 
have ended when pc equals the length of the asm list
'''


def run():
    global pc, STACK_SIZE, label_regex, label_hit_counts
    check_static_rules()
    recursed_labels = set()
    labels = [l for l in asm if (re.match('{}:'.format(label_regex), l))] + list(linked_labels.keys())
    label_hit_counts = dict(zip(labels, [0] * len(labels)))
    while pc < len(asm):
        line = asm[pc]
        # This checks for recursion by determining if the current pc
        # is saved in the link register at the time of a bl instr. If so,
        # this is the 2nd time this bl instr has been reached.
        # Will not detect a recursive procedure if termination condition
        # is immediately met.
        if (re.match('bl {}'.format(label_regex), line)):
            if (pc == reg['lr']):
                # last match is the label
                label = re.findall(label_regex, line)[-1]
                recursed_labels.add(label)

        # check for stack errors
        if (reg['sp'] < 0):
            raise ValueError("stack overflow")
        if (reg['sp'] > STACK_SIZE):
            raise ValueError("stack underflow (make sure to allocate space)")
        if ((reg['sp'] + 1) % 16 != 0):
            raise ValueError("Alignment error: sp must be a multiple of 16")

        # if a label in encountered, inc pc and skip
        # also update label_hit_counts
        if (re.match(label_regex + ':', line)):
            pc += 1;
            label_hit_counts[line] += 1
            continue
        execute(line)
        reg['xzr'] = 0
        pc += 1
    # empty recursed_labels list means no recursion happened
    if (recursed_labels and forbid_recursion):
        raise ValueError("recursion occurred in program but it should not have")
    if (not recursed_labels and require_recursion):
        raise ValueError("recursion did not occur in program but it should have")
    # case where there was recursion, but not for the labels specified
    # in the recursive_labels list. (recursive_labels should be a subset
    # of recursed_labels)
    if (recursed_labels and recursive_labels - recursed_labels):
        raise ValueError("recursive calls do not include required call to {}".format(recursive_labels))


'''
Simple REPL for testing instructions. Limited to instructions that
only affect registers (no memory access or jumps). Prints the flags
and affected registers after executing each instruction.
'''


def repl():
    global n_flag, z_flag, register_regex
    print('armsim repl. operations on memory not supported\ntype q to quit')
    instr = ''
    while (True):
        instr = input('>> ').lower()
        if (instr.startswith('q')): break
        # if enter is pressed with no input skip the rest
        if (not instr): continue
        try:
            execute(instr)
            for r in set(re.findall(register_regex, instr)):
                print("{}: {}".format(r, reg[r]))
            print("Z: {} N: {}".format(n_flag, z_flag))
        except ValueError as e:
            print(e)
    return


'''
A procedure to return the simulator to it's initial state
'''


def reset():
    global reg, z_flag, n_flag, pc
    global require_recursion, forbid_recursion, forbid_loops
    global cycle_count, execute_count
    global ld_cycle, ld_dst
    global flag_cycle, last_dst
    global linked_labels
    forbidden_instructions.clear()
    require_recursion = False
    forbid_recursion = False
    forbid_loops = False
    reg = {r: 0 for r in reg}
    mem.clear()
    asm.clear()
    sym_table.clear()
    n_flag = False;
    z_flag = False
    pc = 0
    cycle_count = 0
    execute_count = 0
    ld_cycle, ld_dst = -1, -1
    flag_cycle = -1
    last_dst = -1
    linked_labels = {}


def main():
    if (not sys.argv[1:]):
        repl()
    else:
        _file = sys.argv[1]
        with open(_file, 'r') as f:
            parse(f.readlines())
        run()

    global cycle_count
    print(f"Total Cycles: {cycle_count}")
    print(f"Executed Instructions: {execute_count}")
    return reg['x0']


if __name__ == "__main__":
    main()
