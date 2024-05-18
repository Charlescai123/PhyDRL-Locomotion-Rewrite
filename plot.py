import pickle
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filepath = f"logs/log_2024_01_31_13_34_12.pkl"   # acc
# filepath = f"logs/log_2024_01_31_13_39_02.pkl"   # phydrl
# filepath = f"logs/log_2024_01_31_13_43_02.pkl"   # phydrl-a1
# filepath = f"logs/log_2024_01_31_14_38_01.pkl"



def normal_read():
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def pd_read():
    return pd.read_pickle(filepath)

if __name__ == '__main__':
    content = normal_read()

    step = []
    desired_speed = []
    speed = []
    for i in range(len(content)):
        step.append(i)
        desired_speed.append(content[i]['desired_speed'][0][0])
        speed.append(content[i]['base_vels_body_frame'])
        # print(content[i]['desired_speed'])
        # print(type(content[i]['desired_speed']))

    step = np.array(step)
    speed = np.array(speed)
    desired_speed = np.array(desired_speed)
    plt.plot(step, desired_speed, speed)
    plt.show()
    # print(content[0])
    # print(content[1])

