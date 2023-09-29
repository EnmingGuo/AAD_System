import subprocess
import os

'''训练集函数'''
def train(num1:float,num2:float,num3:float,filename:list):
    '''
    :param num1: 频繁边:0.15 0.17 0.19 0.21 0.23 0.25
    :param num2: 频繁子图:0.06 0.07 0.08 0.09
    :param num3: 频繁序列:3 4 5 6
    :param filename:
    :return:
    '''
    if os.path.exists('res.txt'):
        os.remove('res.txt')
    fname = "大创项目9-23.exe"

    num1 = float(num1) / 100.0 * (0.16 - 0.15) + 0.15
    num2 = float(num2) / 100.0 * (0.1 - 0.07) + 0.07
    num3 = int(float(num3) / 100.0 * (10 - 3) + 3)

    '''提取患者编号模块'''
    L_id = []
    for name in filename:
        s = ""
        for c in name:
            if c>='0'and c<='9':
                s = s+c
        L_id.append(int(s))
    L_id.sort()
    print(L_id)
    L_tmp = []
    L_seg = []
    for i in range(len(L_id)):
        L_tmp.append(L_id[i])
        if i == len(L_id) - 1 or (L_id[i] <= 89 and i + 1 < len(L_id) and L_id[i + 1] > 89) or (
                i + 1 < len(L_id) and L_id[i + 1] - L_id[i] > 1):
            if len(L_tmp) == 0: continue
            L_seg.append(L_tmp[0] * 4 - 3)
            L_seg.append(L_tmp[-1] * 4)
            if (L_id[i] <= 89 and i + 1 < len(L_id) and L_id[i + 1] > 89) or i == len(L_id) - 1:
                L_seg.append(0)
                L_seg.append(0)
            L_tmp = []
    s = str(num1) + " " + str(num2) + " " + str(num3)
    for i in L_seg:
        s = s + " " + str(i)
    print("训练集信息:", s)
    sorce = s.encode()
    p = subprocess.Popen(fname, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = p.communicate(input=sorce)


'''测试集函数'''
def test(filename:tuple):
    '''提取患者编号模块'''
    L_id = []
    for name in filename:
        s = ""
        for c in name:
            if c >= '0' and c <= '9':
                s = s + c
        L_id.append(int(s))
    L_tmp = []
    L_seg = []
    for i in range(len(L_id)):
        L_tmp.append(L_id[i])
        if i == len(L_id) - 1 or (L_id[i] <= 89 and i + 1 < len(L_id) and L_id[i + 1] > 89) or (
                i + 1 < len(L_id) and L_id[i + 1] - L_id[i] > 1):
            if len(L_tmp) == 0: continue
            L_seg.append(L_tmp[0] * 4 - 3)
            L_seg.append(L_tmp[-1] * 4)
            L_tmp = []
    s = ""
    for i in L_seg:
        s = s + " " + str(i)
    print("测试集信息:", s)
    sorce = s.encode()
    fname = "大创项目-检验-9.23.exe"
    p = subprocess.Popen(fname, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = p.communicate(input=sorce)
    #print(list(map(float,result[0].split())))
    return list(map(float,result[0].split()))

'''filename = ('file1.txt','file2.txt','file3.txt','file88.txt','file89.txt','file90.txt')
train(0.19,0.07,4,filename)
filename = ('file1.txt',)
test(filename)'''