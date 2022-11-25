import numpy as np

def Furness(matrix,cons_sum_rows,cons_sum_cols, EpsilonCriterion, max_steps:int):
  # matrix: matrix to be balanced
  # cons_sum_rows : vector of constraints for the sum of rows
  # cons_sum_cols : vector of constraints for the sum of columns
  # EpsilonCriterion: required precision (stopping criterion)
  # max_steps: maximum number of steps (stopping criterion)
  cons_sum_rows=np.atleast_1d(np.squeeze(cons_sum_rows))# to ensure that the constraints are always 1-dimensional arrays
  cons_sum_cols=np.atleast_1d(np.squeeze(cons_sum_cols))
  matrix=np.atleast_2d(matrix)

  if np.sum(cons_sum_rows) * np.sum(cons_sum_cols) > 0:
    if np.sum(matrix)==0:
      print("the initial matrix contains only zeros, but not the constraints")
    else:
        StepNb = 0
        SumRows = np.sum(matrix,0)
        SumCols = np.sum(matrix,1)
        Nr = np.shape(matrix)[0]
        Nc = np.shape(matrix)[1]
        epsilon = EpsilonCriterion+1 # simply to ensure that we enter the loop.
        a = np.ones(Nr, dtype=np.float32)
        b = np.ones(Nc, dtype=np.float32)
        if max(np.logical_and(SumRows==0,cons_sum_rows>0)):
            print('Problem not solvable because of row constraints')
        if max(np.logical_and(SumCols==0,cons_sum_cols>0)):
            print('Problem not solvable because of column constraints')
        while StepNb <= max_steps and epsilon > EpsilonCriterion:
            epsilon = 0
            StepNb += 1
            a_old = a
            b_old = b
            a = np.divide(cons_sum_cols, np.matmul(matrix, b), out=np.ones_like(cons_sum_cols),where=cons_sum_cols!=0)
            b = np.divide(cons_sum_rows, np.matmul(a, matrix), out=np.ones_like(cons_sum_rows),where=cons_sum_rows!=0)
            epsilon = np.max([np.max(abs(np.divide(a, a_old) - 1)), np.max(abs(np.divide(b, b_old) - 1))])            
        print("epsilon:{0}".format(epsilon))
  return a.reshape((Nr,1)) * matrix * b