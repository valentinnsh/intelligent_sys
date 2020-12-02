# -*- coding: utf-8 -*-
import dimath

n = dimath.matr.fromList([[1.4,-4.88,3000], [1341,233,-12.3], [-200, -47, -34.4]])
m = dimath.matr.fromList([[1,1,1], [0,2,0], [4, 3.5, -5]])
e = dimath.matr.make_identity(3)
print(n)
ni = n.get_inverted()
print(ni)
print(n*ni)
