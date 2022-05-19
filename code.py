import numpy as np
from sklearn.cluster import KMeans
from scipy.io import loadmat, savemat
import time

def load_data_TZ_tongzhi(data_path):
    '''
    加载TZ_同指.txt的数据.
    Arg：
    path：原数据txt文件的路径。
    Return:
    data:包含了所有手指的List，每一个List里面嵌套了以ndarry格式的三元组（X,Y,theta）。
    labels:每个指纹数据对应的人。
    '''
    labels = []
    data = []
    with open(data_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            one_print = []
            line = line.split(', ')
            labels.append(line[0])
            for i in range(int(line[1])):
                one_print.append(np.array(line[2+3*i:5+3*i],dtype=int))
            data.append(one_print)
    return data,labels
class Get_feature:
    '''
    获取指纹中三个细节点构成三角形的特征。
    '''

    def get_anale(self, L1, L2):
        '''
        计算两个边的夹角。
        Args:
        L1,L2:用向量表示的两个边。
        Return: 两条边夹角的度数。
        '''
        Lx = np.sqrt(L1 @ L1)
        Ly = np.sqrt(L2 @ L2)
        cos_angle = L1 @ L2 / (Lx * Ly)
        angle = np.arccos(cos_angle)

        return angle * 360 / 2 / np.pi

    def rotation_dir(self, P1, P2, P3):
        '''
        返回旋转方向:
        Arg:
        p_1,p_2,p_3分别对应的最大点，次大点，最小点 （x,y）
        Return:旋转方向。
        '''
        z21 = P2 - P1
        z32 = P3 - P2
        theta = z21[0] * z32[1] - z32[0] * z21[1]
        if theta < 0:
            theta = -1
        if theta > 0:
            theta = 1
        return theta

    def X_angle(self, P2, P3):
        "最大边与x轴夹角"
        z32 = P3 - P2
        x = np.array([0, 1])
        return self.get_anale(z32, x)

    def get_featrue(self, P1, P2, P3):
        '''
        获得查询所需的特征，包含三角形中的最小角，次小角和最大角对应的边，三角形旋转方向。
        Args:
        P1,P2,P3:三角形三个点的坐标（x，y,a）
        Return:[最小角，次小角,最大角对应的边,三角形旋转方向,P1细节点方向,P2细节点方向,P3细节点方向]
        '''

        # 将坐标和方向分开
        p_1, p_2, p_3 = P1[:2], P2[:2], P3[:2]
        d_1, d_2, d_3 = P1[-1], P2[-1], P3[-1]

        # 获得角度
        a1 = self.get_anale(p_2 - p_1, p_3 - p_1)
        a2 = self.get_anale(p_1 - p_2, p_3 - p_2)
        a3 = self.get_anale(p_1 - p_3, p_2 - p_3)

        angles = [[a1, p_2 - p_3, p_1, d_1],
                  [a2, p_1 - p_3, p_2, d_2],
                  [a3, p_1 - p_2, p_3, d_3]]

        # 按照角度从小到大排序
        angles = sorted(angles, key=lambda x: x[0])

        # 获得旋转方向
        ro_dir = self.rotation_dir(angles[2][2], angles[0][2], angles[1][2])

        # 获得最大边的长度
        max_line = np.sqrt(angles[2][1].dot(angles[2][1]))

        # 获得最大边与X轴的夹角
        max_x_angle = self.X_angle(angles[0][2], angles[1][2])

        return angles[0][0], angles[1][0], max_line, ro_dir, max_x_angle, angles[2][3], angles[0][3], angles[1][3]


class Fingerprint_matching:
    '''
    在面多较多数据时，由于无序的匹配太过耗时，所以我们对模板库的特征进行了有序编排。
    '''

    def __init__(self, base_feature=None, base_label=None):
        if base_feature:
            self.base_feature = self.get_ordered_feature(base_feature)
            self.base_label = base_label

    def save(self, path):
        savemat(path, {'feature': self.base_feature, 'label': self.base_label})
        print('保存成功！')

    def load(self, path):
        data = loadmat(path)
        # print(data)
        self.base_feature = data['feature']
        self.base_label = data['label']

    def get_ordered_feature(self, base_data):
        # 开始重新修改特征
        print('开始重新修改特征!')
        new_feature = []
        for d in base_data:
            # print(len(fp))
            _, feature = triangle(d)
            feature = sorted(feature, key=lambda x: x[0])
            ordered_feature = []
            ordered_feature.append([i for i in feature if i[3] == -1])
            ordered_feature.append([i for i in feature if i[3] == 1])
            new_feature.append(ordered_feature)

        return new_feature

    def is_one(self, P, T):
        '''
        检索两个指纹的匹配度,输入两个指纹对应的所有三角特征。
        Arg:
            T,P:模版指纹，和需要匹配的指纹。
        Return:包含每两个相似的匹配信息:[i,j,simi,Angs,Roan]，其长度为匹配三角形的个数。
        '''

        records = []  # 用以存放匹配的三角形id以及一些相关信息
        for i in range(len(P)):
            tr = 0 if P[i][3] == -1 else 1

            for j in range(len(T[tr])):

                "第二个判断计算两个三角形 和 的最小角、 次小角和最大边长的差值"
                if T[tr][j][0] - P[i][0] > 3:
                    break

                deta1 = abs(T[tr][j][0] - P[i][0])
                deta2 = abs(T[tr][j][1] - P[i][1])
                deta3 = abs(T[tr][j][2] - P[i][2])

                if deta1 <= 3 and deta2 <= 4 and deta3 <= 5:
                    "计算这两个三角形 的结构相似度"
                    simi = ((3 - deta1) / 3 + (4 - deta2) / 4 + (5 - deta3) / 5) / 3
                else:
                    continue  # 返回到第一个判断

                "第三个判断脊线方向"
                ang_1 = T[tr][j][5] - P[i][5]
                ang_2 = T[tr][j][6] - P[i][6]
                ang_3 = T[tr][j][7] - P[i][7]

                Dang_1 = abs(ang_1 - ang_2)
                Dang_2 = abs(ang_1 - ang_3)
                Dang_3 = abs(ang_2 - ang_3)
                if Dang_1 <= 15 and Dang_2 <= 15 and Dang_3 <= 15:
                    Angs = 3
                elif (ang_1 <= 30 and Dang_2 <= 15 and Dang_3 <= 15) or (
                        ang_1 <= 15 and Dang_2 <= 30 and Dang_3 <= 15) or (
                        ang_1 <= 15 and Dang_2 <= 15 and Dang_3 <= 30):
                    Angs = 2
                else:
                    continue  # 返回到第一个判断

                "计算两个三角形最大边与X轴角度差"
                Roan = T[tr][j][4] - P[i][4]
                record = [i, j, simi, Angs, Roan]
                records.append(record)

        return records

    def match(self, input):
        '''
        使用输入特征和模板中的进行匹配。
        '''
        print('开始匹配！')
        _, input = triangle(input)
        points_num = []
        max_v = 0
        # max_name = None
        a = time.time()
        for feature, label in zip(self.base_feature, self.base_label):
            record = self.is_one(input, feature)
            # clu_n = max_clu_num(record)
            clu_n = len(record)
            if clu_n > max_v:
                max_v = clu_n
                max_name = label
            points_num.append(clu_n)

        b = time.time()
        print('time:', b - a)
        points_num = np.array(points_num)
        print(points_num.shape)
        index = np.argsort(points_num, axis=0)
        index = index[::-1]
        # points_num = points_num[index]

        return self.base_label[index[:2080]], points_num[index[:2080]]


def triangle(data):
    '''
    获得指纹中所有满足条件的三角形的顶点坐标和特征。
    Args:
        data:包含单个指纹所有细节点和方向的列表。
    Return:
        sanjiao:所有三角形的坐标。
        all_feature:所有的特征。
    '''
    sanjiao = []
    all_feture = []
    for i in range(len(data) - 2):
        for j in range(i + 1, len(data) - 1):
            for k in range(j + 1, len(data)):
                feature = Get_feature()
                min_angle, mid_angle, max_line, ro_dir, max_x_angle, p1_angle, p2_angle, p3_angle = feature.get_featrue(
                    data[i], data[j], data[k])
                if min_angle >= 10 and min_angle <= 25:
                    if mid_angle >= min_angle and mid_angle <= 45:
                        if max_line >= 30 and max_line <= 80:
                            sanjiao.append([data[i], data[j], data[k]])
                            feature = [min_angle, mid_angle, max_line, ro_dir, max_x_angle, p1_angle, p2_angle,
                                       p3_angle]
                            all_feture.append(feature)

    return sanjiao, all_feture


def is_one(T, P):
    '''
    检索两个指纹的匹配度,输入两个指纹对应的所有三角特征。
    Arg:
        T,P:模版指纹，和需要匹配的指纹。
    Return:包含每两个相似的匹配信息:[i,j,simi,Angs,Roan]，其长度为匹配三角形的个数。
    '''

    records = []  # 用以存放匹配的三角形id以及一些相关信息
    for i in range(len(T)):
        for j in range(len(P)):
            "第一个判断旋转角"
            if T[i][3] != P[j][3]:
                continue
            "第二个判断计算两个三角形 和 的最小角、 次小角和最大边长的差值"
            deta1 = abs(T[i][0] - P[j][0])
            deta2 = abs(T[i][1] - P[j][1])
            deta3 = abs(T[i][2] - P[j][2])
            if deta1 <= 3 and deta2 <= 4 and deta3 <= 5:
                "计算这两个三角形 的结构相似度"
                simi = ((3 - deta1) / 3 + (4 - deta2) / 4 + (5 - deta3) / 5) / 3
            else:
                continue  # 返回到第一个判断

            "第三个判断脊线方向"
            ang_1 = T[i][5] - P[j][5]
            ang_2 = T[i][6] - P[j][6]
            ang_3 = T[i][7] - P[j][7]
            Dang_1 = abs(ang_1 - ang_2)
            Dang_2 = abs(ang_1 - ang_3)
            Dang_3 = abs(ang_2 - ang_3)
            if Dang_1 <= 15 and Dang_2 <= 15 and Dang_3 <= 15:
                Angs = 3
            elif (ang_1 <= 30 and Dang_2 <= 15 and Dang_3 <= 15) or (ang_1 <= 15 and Dang_2 <= 30 and Dang_3 <= 15) or (
                    ang_1 <= 15 and Dang_2 <= 15 and Dang_3 <= 30):
                Angs = 2
            else:
                continue  # 返回到第一个判断

            "计算两个三角形最大边与X轴角度差"
            Roan = T[i][4] - P[j][4]
            record = [i, j, simi, Angs, Roan]
            records.append(record)

    return records


def max_clu_num(record, c=4):
    '''
    根据聚类结果中找出个数最多的类的个数。
    Arg:
        record:从triangle中得到的包含roans的个数结果。
        c:聚类的个数。
    Return:经过聚类后的相似三角形的个数。
    '''
    roans = np.array([i[4] for i in record]).reshape(-1, 1)
    if len(roans) <= c:
        return len(roans)
    km = KMeans(c)
    km.fit(roans)
    label = km.labels_
    li = []
    for i in range(c):
        li.append(np.sum(label == i))
    return max(li)


def test(target_feature, base_feature, base_label):
    '''
    根据输入的指纹特征，从指纹库中找出匹配的人的名字。
    Args:
        target_feature:待匹配的指纹的特征。
        base_feature:指纹库中的所有指纹的特征。
        base_label:指纹库中指纹特征对应的人的名字。
    Return:
        points_num:待破匹配指纹与库中所有指纹之间的相似三角形的个数。
        max_name:拥有最多相似三角形的指纹库中对应指纹的人的名字。
    '''
    points_num = []
    max_v = 0
    max_name = None

    for feature, label in zip(base_feature, base_label):
        _, target_tri = triangle(target_feature)
        _, base_tri = triangle(feature)
        result = is_one(target_tri, base_tri)
        max_num = max_clu_num(result)
        if max_num > max_v:
            max_v = max_num
            max_name = label
        points_num.append(max_num)
    return points_num, max_name


if __name__ == '__main__':


    luanzhi_path = '数据/TZ_同指200_乱序后_Data.txt'
    yizhi_path = '数据/TZ_异指.txt'       #路径名字

    luanzhi_data, luanzhi_label = load_data_TZ_tongzhi(luanzhi_path)

    yizhi_data, yizhi_label = load_data_TZ_tongzhi(yizhi_path)

    base_data = luanzhi_data + yizhi_data
    base_label = luanzhi_label + yizhi_label
    print('base_len:', len(base_label))

    FB = Fingerprint_matching(base_data, base_label)
    FB.save('./base.mat')                          #保存提取的三角形特征

    # name,li= FB.match(target_featur)
    # FB = Fingerprint_matching()
    # FB.load('./base.mat')
    k = 2080                  #可以修改，过滤得到图像数
    "保存检索结果"
    for i in range(len(luanzhi_data)):
        f = open('./result1/{}.txt'.format(luanzhi_label[i]), 'w')

        f2 = open('./result2/match_number{}.txt'.format(luanzhi_label[i]), 'w')
        name, li = FB.match(luanzhi_data[i])
        f.write(str(k) + ',')
        for i in range(k):
            f.write(name[i] + ',')
            f2.write(str(li[i]) + ',')
        f.close()
        f2.close()

