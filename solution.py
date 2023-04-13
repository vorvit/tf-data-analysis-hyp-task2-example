import pandas as pd
import numpy as np
from scipy import stats

chat_id = 1056349463 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    
    res = stats.cramervonmises_2samp(x, y)
    return res.pvalue < 0.01
