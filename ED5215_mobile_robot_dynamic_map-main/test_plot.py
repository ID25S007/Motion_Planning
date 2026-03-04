import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    plt.plot([1, 2, 3])
    plt.savefig('test_plot.png')
    print('Plot saved successfully')
except Exception as e:
    print(f'Error: {e}')
