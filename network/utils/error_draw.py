import numpy as np
import pandas as pd

errors = np.load('test_errors.npy')
errors_name = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']

data_df = pd.DataFrame(errors, columns=errors_name, index=[i for i in range(697)])


writer = pd.ExcelWriter('test_errors.xlsx')
data_df.to_excel(writer, 'page_1')
writer.save()


print('haha')