# NMF-Gradient

Разложение матрицы предпочтений производится с помощью NMF и метода проекции градиента и подбора  \alfa_t, описанного в статье http://www.csie.ntu.edu.tw/~cjlin/papers/pgradnmf.pdf
Минимизируется норма Фробениуса разницы исходной матрицы и ее аппроксимации. 

#based on scikit-learn

В качестве данных для матрицы предпочтений используется MovieLens 1M Dataset http://grouplens.org/datasets/movielens/.
