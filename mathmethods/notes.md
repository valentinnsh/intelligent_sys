# Что тут есть?
## Представление данных
Матрица хранится как list строк. Методы доступа по индексу переопределены, поэтому запись вида `a[1][3]` применима.

## Готовые на данный момент методы
- Переопределены операторы `+, -, *, /` , причем в случае когда второым операндом выступает число операется для матрицы выполняется поэлементно.
- Операции со строками `row_add, row_div, row_mul, row_sub(row number, value)` меняют значения в текущей матрице, в то время как эти же методы но с припиской `get_` возвращают новую матрицу с измененной строкой.
- Среднее значение по столбцам матрицы `unimean`
- Добавление единиц справа и снизу `add_ones`
- Транспонирование `transpose` меняет текущую матрицу на транспонированную, в то время как `get_transpose` возвращает транспонированную матрицу, не меняя оригинал.
- `get_rank` возвращает размер матрицы
- далее

## Прочее
- `fromList` создает матрицу из листа листов
- `make_identity(n)` создает единичную матрицу nxn
- `make_zero(n)` создает нулевую матрицу nxn
- `make_random(m,n,lowval,highval)` создает случайную матрицу