import armsim
#run instruction tests

'''
Full program tests in /tests directory
exit code in x0 for all test cases should be 7
'''
with open('tests/arithmetic_test.s', 'r') as f:
	armsim.parse(f.readlines())
armsim.run()
assert armsim.reg['x0'] == 7, "arithmetic_test returned incorrect value of {}".format(armsim.reg['x0'])
armsim.reset()
with open('tests/branch_test.s', 'r') as f:
	armsim.parse(f.readlines())
armsim.run()
assert armsim.reg['x0'] == 7, "arithmetic_test returned incorrect value of {}".format(armsim.reg['x0'])
armsim.reset()

with open('tests/brk_test.s', 'r') as f:
	armsim.parse(f.readlines())
armsim.run()
assert armsim.reg['x0'] == 7, "brk_test returned incorrect value of {}".format(armsim.reg['x0'])
armsim.reset()

''' Test Load/Store '''
with open('tests/load_store_test.s', 'r') as f:
	armsim.parse(f.readlines())

armsim.run()
assert armsim.reg['x1'] == 189
assert armsim.reg['x2'] == -67
assert armsim.reg['x3'] == 65254
assert armsim.reg['x4'] == -282
assert armsim.reg['x5'] == -100000
assert armsim.reg['x6'] == -88
assert armsim.reg['x7'] == 168
assert armsim.reg['x9'] == -88
assert armsim.reg['x10'] == 65448
armsim.reset()

print("All tests passed")  

test = [5, 10, 15]
print(test.index(0))