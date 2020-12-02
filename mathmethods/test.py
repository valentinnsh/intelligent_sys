import dimath

n = dimath.matr.fromList([[1,0,0], [1,2,3], [8, 3, 4]])
m = dimath.matr.fromList([[1,1,1], [0,2,0], [4, 3.5, -5]])
e = dimath.matr.make_identity(3)
print(n)
print(n.get_inverted())
print(zip(range(3), range(3)))
