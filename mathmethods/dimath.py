# -*- coding: utf-8 -*-
# V.Shishkin 2020
#
#


import operator
import sys
import pickle
import random

class matr(object):

    def __init__(self, m, n, init=True):
        if init:
            self.rows = [[0]*n for x in range(m)]
        else:
            self.rows = []
        self.m = m
        self.n = n

    # getitem TODO
    def __getitem__(self, idx):
        return self.rows[idx]

    # по индексу
    def __setitem__(self, idx, item):
        self.rows[idx] = item

    def __str__(self):
        s='\n'.join([' '.join([str(item) for item in row]) for row in self.rows])
        return s + '\n'

    def transpose(self):
        """ Transpose the matrix. Changes the current matrix """

        self.m, self.n = self.n, self.m
        self.rows = [list(item) for item in zip(*self.rows)]

    def get_transpose(self):
        """ Return a transpose of the matrix without
        modifying the matrix itself """

        m, n = self.n, self.m
        mat = matr(m, n)
        mat.rows =  [list(item) for item in zip(*self.rows)]

        return mat

    def get_rank(self):
        return (self.m, self.n)

    def __eq__(self, mat):
        return (mat.rows == self.rows)

    def __add__(self, mat):
        # Случай добавления числа
        if isinstance(mat, matr) == False:
            res = matr(self.m, self.n)

            for x in range(self.m):
                row = [i+mat for i in self.rows[x]]
                res[x] = row

            return res

        res = matr(self.m, self.n)

        for x in range(self.m):
            row = [sum(i) for i in zip(self.rows[x], mat[x])]
            res[x] = row

        return res

    def __sub__(self, mat):
        # Случай добавления числа
        if isinstance(mat, matr) == False:
            res = matr(self.m, self.n)

            for x in range(self.m):
                row = [i-mat for i in self.rows[x]]
                res[x] = row

            return res

        res = matr(self.m, self.n)

        for x in range(self.m):
            row = [i[0]-i[1] for i in zip(self.rows[x], mat[x])]
            res[x] = row

        return res

    # Предполагаем что единственное деление которое нам нужно будет
    # производить это деление матрицы на число
    def __truediv__(self, num):
        res = matr(self.m, self.n)

        for x in range(self.m):
            row = [i-mat for i in self.rows[x]]
            res[x] = row

        return res

    # TODO - faster multiplication mabe?
    def __mul__(self, mat):
        matm, matn = mat.get_rank()
        # Случай умножения на число
        if isinstance(mat, matr) == False:
            res = matr(self.m, self.n)

            for x in range(self.m):
                row = [i-mat for i in self.rows[x]]
                res[x] = row

            return res
        # Матрица на матрицу
        tmpmat = mat.get_transpose()
        matmul = matr(self.m, matn)

        for x in range(self.m):
            for y in range(tmpmat.m):
                matmul[x][y] = sum([item[0]*item[1] for item in zip(self.rows[x], tmpmat[y])])

        return matmul

    #
    # Операции со строками. в двух вариациях с изменением рабочей матрицы и без
    #

    def row_add(self, row_num, val):
        self.rows[row_num] = [self.rows[row_num][i]+val for i in range(self.n)]
    def row_div(self, row_num, val):
        self.rows[row_num] = [self.rows[row_num][i]/val for i in range(self.n)]
    def row_mul(self, row_num, val):
        self.rows[row_num] = [self.rows[row_num][i]*val for i in range(self.n)]
    def row_sub(self, row_num, val):
        self.rows[row_num] = [self.rows[row_num][i]-val for i in range(self.n)]

    def get_row_add(self, row_num, val):
        mat = self
        mat.rows[row_num] = [self.rows[row_num][i]+val for i in range(self.n)]
        return mat
    def get_row_div(self, row_num, val):
        mat = self
        mat.rows[row_num] = [self.rows[row_num][i]/val for i in range(self.n)]
        return mat
    def get_row_mul(self, row_num, val):
        mat = self
        mat.rows[row_num] = [self.rows[row_num][i]*val for i in range(self.n)]
        return mat
    def get_row_sub(self, row_num, val):
        mat = self
        mat.rows[row_num] = [self.rows[row_num][i]-val for i in range(self.n)]
        return mat

    # Среднее значение по солбцам матрицы
    # cls -> self?
    def unimean(cls):
        m = len(cls.rows)
        n = len(cls.rows[0])

        mat = matr(m, n)
        mat.rows =  [list(item) for item in zip(*cls.rows)]

        uni = [sum(i)/float(m) for i in mat.rows]
        #for i in range(m):
         #   uni.append(sum(rows[i])/n)

        return uni

    # Добавление солбца и строки единиц справа и снизу соответственно
    def add_ones(self):
        for i in self.rows:
            i.append(1)
        self.rows.append([1]*self.m)

    # Поиск обратной матрицы методом Гаусса
    def get_inverted(clc):
        m = len(clc.rows)
        n = len(clc.rows[0])

        a = matr(m,n*2)

        a.rows = [i+[0]*n for i in clc.rows]
        for i in range(n):
            a[i][n+i] = 1

        #Applying Guass Jordan Elimination
        for i in range(n):
            if a[i][i] == 0.0:
                sys.exit('Divide by zero detected!')

            for j in range(n):
                if i != j:
                    ratio = a[j][i]/float(a[i][i])
                    for k in range(2*n):
                        a[j][k] = a[j][k] - ratio * a[i][k]
        # Row operation to make principal diagonal element to 1
        for i in range(n):
            divisor = a[i][i]
            for j in range(2*n):
                a[i][j] = a[i][j]/float(divisor)


        inverted = matr(m,n)
        inverted.rows = [i[n:] for i in a.rows]

        return inverted

    # Add another way of saving
    def save_obj(cls, name ):
        with open('obj/'+ name + '.pkl', 'wb') as f:
            pickle.dump(cls, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    # Redo load mechanism
    def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    # def loat_to_cls(name):

    @classmethod
    def _makematr(cls, rows):

        m = len(rows)
        n = len(rows[0])

        mat = matr(m,n, init=False)
        mat.rows = rows

        return mat

    @classmethod
    # случайная матрица получаемая с использованием random
    def make_random(cls, m, n, low=0, high=10):

        obj = matr(m, n, init=False)
        for x in range(m):
            obj.rows.append([random.randrange(low, high) for i in range(obj.n)])

        return obj

    @classmethod
    def make_zero(cls, m, n):
        rows = [[0]*n for x in range(m)]
        return cls.fromList(rows)

    @classmethod
    # создает единичную матрицу размера mxm
    def make_identity(cls, m):
        rows = [[0]*m for x in range(m)]
        idx = 0

        for row in rows:
            row[idx] = 1
            idx += 1

        return cls.fromList(rows)

    @classmethod
    # для создания матрицы из листа листов
    def fromList(cls, listoflists):
        # E.g: matr.fromList([[1 2 3], [4,5,6], [7,8,9]])

        rows = listoflists[:]
        return cls._makematr(rows)
