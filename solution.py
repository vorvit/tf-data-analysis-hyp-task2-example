import pandas as pd
import numpy as np
from scipy import stats
from hyppo.ksample import MMD

chat_id = 1056349463 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    
    res = MMD(compute_kernel = "rbf", gamma = 1).test(x, y)[1]
#   res = stats.anderson_ksamp([x, y])
#   res = stats.cramervonmises_2samp(x, y)
#   return res.pvalue < 0.01
    return res < 0.01
