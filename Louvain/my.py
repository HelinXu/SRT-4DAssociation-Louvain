from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import array

import numbers
import warnings

import networkx as nx
import numpy as np
import cv2
import math

__PASS_MAX = -1
__MIN = 0.0000001


class Status(object):
    """
    To handle several data in one struct.

    Could be replaced by named tuple, but don't want to depend on python 2.6
    """
    node2com = {}  # node 2 community 的字典，partition的返回值。
    total_weight = 0  # 初始化：Return the number of edges or total of all edge weights. 初始化后不做更改。
    internals = {}  # community超节点内部自己与自己相连的权重之和，字典类型
    degrees = {}  # 下标（key）为community编号
    gdegrees = {}  # node 2 degree

    def __init__(self):
        self.node2com = dict([])  # node 2 community 的字典，partition的返回值。
        self.total_weight = 0
        self.degrees = dict([])  # 下标（key）为community编号
        self.gdegrees = dict([])  # node 2 degree
        self.internals = dict([])  # community超节点内部自己与自己相连的权重之和，字典类型
        self.loops = dict([])  # node 2 loop_weight_float

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        self.node2com = dict([])
        self.total_weight = 0
        self.degrees = dict([])
        self.gdegrees = dict([])
        self.internals = dict([])
        self.total_weight = graph.size(weight=weight)  # Return the number of edges or total of all edge weights.
        if part is None:
            # 对于从头开始的构建的情况
            for node in graph.nodes():   # node，node形成的list 遍历list中每一个node
                self.node2com[node] = count   # community的编号 计数
                deg = float(graph.degree(node, weight=weight))   # 对于现在遍历到的这个node，取它的度 float，为边权和
                if deg < 0:
                    error = "Bad node degree ({})".format(deg)
                    raise ValueError(error)
                self.degrees[count] = deg   # count-当前community编号。给degree词典初始化对应community的度（com-degree）
                self.gdegrees[node] = deg   # key为节点。初始化节点-节点的度的词典。
                # This is identical to G[u][v] except the default is returned
                # instead of an exception is the edge doesn't exist.
                edge_data = graph.get_edge_data(
                    node, node, default={weight: 0})  # 自己节点到自己本身的loop连边（字典类型），默认为{weight: 0}
                # dict.get(key, default=None)
                # default -- 如果指定键的值不存在时，返回该默认值1。返回指定键的值，如果键不在字典中返回默认值 None 或者设置的默认值。
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]  # community超节点内部自己与自己相连的权重之和，字典类型
                count += 1  # 计数，community编号
        else:
            # 目前还没有使用这种情况
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):  # 读入返回都是 dendo 或 partition （node 2 community 的字典）TODO
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])  # community 2 community内部边权和
    deg = dict([])
    links = graph.size(weight=weight)  # Return the number of edges or total of all edge weights.
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph.nodes():  # 这句本来是for node in graph:
        com = partition[node]   # node 2 community
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)  # community的度
        for neighbor, datas in graph.nodes.items():  # Python 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组。
            # 上一句本来是for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:  # delta
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)  # 自环
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.  # 连边，community内部。因为这样的边自动会被计算两遍。

    res = 0.
    for com in set(partition.values()):
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None):
    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state)  # 类型就是 partition 是一个树状图，每一层是一个partition（字典）
    return partition_at_level(dendo, 1)


def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link 特殊情况：每个点都在一个独立的社区，这种情况在我们的项目中不应该出现。
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)  # part_init是初始partition，默认用None就可以。weight是边权重字典的key，默认就是“weight”
    status_list = list()
    # 如果需要让函数修改某些数据，则可以通过把这些数据包装成列表、字典等可变对象，
    # 然后把列表、字典等可变对象作为参数传入函数，在函数中通过列表、字典的方法修改它们，这样才能改变这些数据。
    __one_level(current_graph, status, weight, resolution, random_state)
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)  # 将node对应的community编号规范化成从0开始的整数列
    status_list.append(partition)
    mod = new_mod
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution, random_state)
        new_mod = __modularity(status, resolution)
        if new_mod - mod < __MIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        current_graph = induced_graph(partition, current_graph, weight)  # 得到了社区为超级节点的各个节点之间连接的总权重图
        status.init(current_graph, weight)
    return status_list[:]  # 语法上是否有[:]好像没有关系，status_list是一个list，里面是一系列partition


def induced_graph(partition, graph, weight="weight"):  # partition：node 2 com
    ret = nx.Graph()
    ret.add_nodes_from(partition.values())  # 节点都是community的序号

    for node1, node2, datas in graph.edges(data=True):  # node-点。partition[node（key）] - community（value）
        edge_weight = datas.get(weight, 1)  # 该边的权重
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)  # 大括号是字典，默认参数，如果未找到，就返回大括号里的字典。
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})  # 每条边是一个字典。此处**表示添加一个边的信息的字典。是更新还是添加？

    return ret  # 得到了社区为节点的各个节点之间连接的总权重图


def __renumber(dictionary):  # status.node2com
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())  # com
    target = set(range(len(values)))  # 将com编号改成从0开始的连续整数列。

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm
    """
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution, random_state):  # 传入的graph已经是以community编号为超级节点的图了！
    """Compute one level of communities 把两个最好的超级节点合并 不对，好像不是两个：双层循环已经把所有可以合并的点都合并完了，下一级是
    社区级别（超节点）的合并了！
    """
    modified = True
    nb_pass_done = 0  # ？
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != __PASS_MAX:  # 如果上一轮有做更改，且总共循环次数没超过int上限
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):  # 随机循环遍历所有点（其实是所有的community）
            com_node = status.node2com[node]  # 该node所在的社区
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)  # NOQA
            neigh_communities = __neighcom(node, graph, status, weight_key)  # @TODO
            remove_cost = - resolution * neigh_communities.get(com_node, 0) + \
                          (status.degrees.get(com_node, 0.) -
                           status.gdegrees.get(node, 0.)) * degc_totw
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in __randomize(neigh_communities.items(), random_state):
                incr = remove_cost + resolution * dnc - \
                       status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:  # 把当前节点放在哪个社区中最好
                modified = True
        new_mod = __modularity(status, resolution)
        if new_mod - cur_mod < __MIN:
            break


def __neighcom(node, graph, status, weight_key):
    """
    Compute the communities in the neighborhood of node in the graph given
    with the decomposition node2com
    """
    weights = {}
    for neighbor, datas in graph[node].items():  # 遍历了哪些点？
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights   # community 2 weights


def __remove(node, com, weight, status):
    """ Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """ Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):
    """
    Fast compute the modularity of the partition of the graph using
    status precomputed
    """
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):   # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / \
                      links - ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items


def build_graph():
    # just for test
    G = nx.Graph()
    f = open("demofile.txt", "r")
    dot_num = int(f.readline())
    for i in range(dot_num):  # 添加顶点
        # print(f.readline())
        G.add_node(int(f.readline()))
    for line in f.readlines():  # 处理联结边
        # print(line)
        list = line.split()
        G.add_edge(int(list[0]), int(list[1]), weight=float(list[2]))
        # print(int(list[0]), int(list[1]), float(list[2]))
        # print(float(G.degree(1, weight='weight')))
    return G


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def get_color(idx, maxIdx):  # 随便搞了个哈希
    r = math.cos((2.0 / 3.0 * math.pi + 179.13 * idx / (maxIdx + 0.12))) * 256
    g = math.cos((2.0 / 3.0 * math.pi + 101.19 * idx / (maxIdx + 0.57))) * 256
    b = math.cos(133.51 * idx / (maxIdx + 0.34) * math.pi) * 256
    return r, g, b


def build_from_4d(frameIdx):
    G = nx.Graph()
    f = open("../output/txt/frame" + str(frameIdx) + ".txt", "r")
    f.readline()  # J OverallIdx viewIdx joint种类0-18(jointIdx) 该joint在图中的顺序(candidix) x y score
    while True:  # 读入并添加所有的点
        line = f.readline().split()
        if line[0] == 'P':
            break
        G.add_node(int(line[0]),
                   viewIdx=int(line[1]),
                   jointIdx=int(line[2]),
                   candidIdx=int(line[3]),
                   x=float(line[4]),
                   y=float(line[5]),
                   score=float(line[6]))
    while True:  # 读入并添加所有的边
        line = f.readline().split()
        if line[0] == 'E':
            break
        G.add_edge(int(line[2]),
                   int(line[3]),
                   viewIdx=int(line[0]),
                   pafIdx=int(line[1]),
                   weight=float(line[6])*4)
    while True:
        # 读入epiEdges: jointIdx viewA viewB aOverAllIdx bOverallIdx score
        line = f.readline().split()
        if line[0] == 'end':
            break
        G.add_edge(int(line[3]),
                   int(line[4]),
                   weight=max(0.0, float(line[5])))  # 暂时只加了这么多参数 注意louvain原始算法不允许-1
    return G


def main():
    capture = []  # 这个数据类型对了吗
    img = []
    node_img = [[] for i in range(5)]  # 这个用来从networkx画点
    frameIdx = 0
    for i in range(5):
        capture.append(cv2.VideoCapture("../data/shelf/video/" + str(i) + ".mp4"))
        img.append(None)  # 需要初始化开辟空间
    f_h = int(capture[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    # total_frame = capture[0].get(cv2.CAP_PROP_FRAME_COUNT)  # 视频的总帧数
    # for frameIdx in range(min(30, total_frame)):

    while True:
        node_list = [[] for i in range(5)]  # 这个用来从networkx画点
        G = build_from_4d(frameIdx)  # frameIdx
        # compute the best partition
        partition = best_partition(G, resolution=10.0)  # 返回一个字典：node 2 community
        # draw the graph
        # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        maxCommuIdx = max(partition.values())
        pos = {}
        for node, data in G.nodes.items():
            pos[node] = (data.get('x') + f_w * (data.get('viewIdx') % 3),
                         -data.get('y') - f_h * int(data.get('viewIdx') / 3))  # 这里正负号好像错了
            data['value'] = partition[node]
            node_list[data.get("viewIdx")].append(data)

        for viewIdx in range(5):
            ret, img[viewIdx] = capture[viewIdx].read()
            if not ret:
                break  # 当获取完最后一帧就结束
            # node_img[viewIdx] = np.zeros([512, 512, 3], np.uint8)  # 创建一副黑色的图片
            for data in node_list[viewIdx]:
                # cv2.circle(node_img[viewIdx], center=(int(data.get('x')), int(data.get('y'))), radius=2, color=(100,100,100))
                # 视频的帧率FPS https://blog.csdn.net/learn_learn_/article/details/112007757
                cv2.circle(img[viewIdx], center=(int(data.get('x')), int(data.get('y'))), radius=5,
                           color=get_color(data.get('value'), maxCommuIdx), thickness=-1)
                cv2.putText(img[viewIdx], text=str(data.get('value')), org=(int(data.get('x')), int(data.get('y'))),
                            color=(256, 256, 256), fontScale=0.3, fontFace=cv2.FONT_HERSHEY_TRIPLEX, thickness=1)
        frame = np.zeros((f_h * 2, f_w * 3, 3), np.uint8)
        frame[0:f_h, 0:f_w] = img[0]
        frame[0:f_h, f_w:2 * f_w] = img[1]
        frame[0:f_h, 2 * f_w:3 * f_w] = img[2]
        frame[f_h:2*f_h, 0:f_w] = img[3]
        frame[f_h:2*f_h, f_w:2*f_w] = img[4]
        cv2.imwrite('../output/tmp/' + str(frameIdx) + '.jpg', frame)  # 存储为图像
        frameIdx += 1
        # node_list.clear()
        if not ret:
            break
        # color the nodes according to their partition
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)  # 以列表返回字典里的所有值:所属的community编号
        nx.draw_networkx_nodes(G, pos, G.nodes(), node_size=40,
                               cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=[float(d['weight']) for (u, v, d) in G.edges(data=True)])
        plt.savefig("../output/tmp/Graph" + str(frameIdx) + '.jpg', format="JPG")
        # networkx 存文件 https://www.coder.work/article/361314
        plt.show()





if __name__ == '__main__':
    main()