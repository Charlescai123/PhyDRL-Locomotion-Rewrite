import re
import numpy as np


def text_value_mean(filename, sentence: str):
    with open(filename, 'r') as f:
        content = f.read()
    lines = content.splitlines()
    value_list = []
    cnt = 0
    s = 0
    for line in lines:
        if sentence in line:
            #num = float(re.findall(r"\d+\.?\d*", line)[0])
            num = eval(line.split(':')[-1])
            value_list.append(num)
            s += float(num)
            cnt += 1
    return value_list


if __name__ == '__main__':
    filename = ("../PhyDRL-Locomotion-Rewrite/log.txt")
    #filename = ("./log.txt")
    sentence = "get action duration:"
    #sentence = "Get PhyDRL action"
    #sentence = "PhyDRL time:"
    #sentence = "total get_action time:"
    #sentence = "get state vector time:"
    #sentence = "stance_action time"
    #sentence = "get drl action time:"

    value_list = text_value_mean(filename=filename, sentence=sentence)
    #print(value_list)
    print(np.mean(value_list[50:-1]))
