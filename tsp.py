

import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt


# 26个字母
CHARACTERS = [chr(i) for i in np.arange(65,91)]
# print(CHARACTERS)

# using to generate graph edge data
# 思路大概如下：①产生边和边权，用networkx画出来
def generate_data(seed,node_num=5,max_weight=20):
    """

    :param seed: 设置生成数据的随机种子
    :param node_num: 设置需要的结点数目
    :param max_weight: 设置边权的最大值
    :return:
    """
    assert node_num <= len(CHARACTERS),"the node num must less than 26!"
    characters = CHARACTERS[:node_num]
    edge_list = list(combinations(characters,2))
    np.random.seed(seed)
    edge_weight = list(np.random.randint(1,max_weight,(len(edge_list))))

    edge_list=list(map(list,zip(*edge_list)))
    edge_weight=[{"weight":i} for i in edge_weight]

    edge_list_0 = edge_list[0]
    edge_list_1 = edge_list[1]
    edge_data=list(zip(edge_list_0,edge_list_1,edge_weight))  # element: (start_node, end_node, weight_value)
                                                              # such as: ('A', 'B', {'weight': 1})
    return edge_data
class nearestNeighbor:
    def __init__(self,G:nx.Graph,seed):
        self.Graph = G
        self.seed = seed
        self.nodes = list(G.nodes())
        self.hamiltion_graph = nx.DiGraph()
        self.cur_node_num=0



    def generate_hamilton(self,):
        # ①随机选取结点 ②开始找最小权值的边，加点 ③直至添加完所有点 ④补上最后一条边构成哈密顿回路

        np.random.seed(self.seed)
        random_idx = np.random.randint(0, len(self.nodes))
        ran_start_node = self.nodes[random_idx]
        self.hamiltion_graph.add_node(ran_start_node)
        self.cur_node_num += 1

        nodes_num = len(self.nodes)

        cur_node_list = list(self.hamiltion_graph.nodes)
        while self.cur_node_num< nodes_num:

           print("current node list:",cur_node_list)

           weight_list=self.Graph.edges(cur_node_list[-1],data='weight')

           sorted_weight_list = sorted(weight_list,key= lambda t:t[2])
           sorted_weight_list=[i for i in sorted_weight_list if i[1] not in cur_node_list ]
           # print("weight_list", sorted_weight_list)
           start,end,weight=sorted_weight_list[0]  #(start,end,weight) min weight


           self.hamiltion_graph.add_edge(start,end,weight=weight)

           self.cur_node_num+=1
           cur_node_list = list(self.hamiltion_graph.nodes)
        print("current node list:", cur_node_list)
        start = cur_node_list[-1]
        end = cur_node_list[0]
        last_edge_weight = self.Graph[start][end]['weight']
        self.hamiltion_graph.add_edge(start,end,weight=last_edge_weight)

        self.print_hamilton_weight_sum()
        return self.hamiltion_graph

    def print_hamilton_weight_sum(self):

        cur_node_list = list(self.hamiltion_graph.nodes)
        cur_node_list.append(cur_node_list[0])
        edge_data = self.hamiltion_graph.edges(data='weight')

        data_dict = {(u, v): d for u, v, d in edge_data}

        path_str=''
        weight_str= ''
        for i in range(len(cur_node_list)-1):

            start=cur_node_list[i]
            end = cur_node_list[i+1]
            weight = data_dict[(start,end)]

            path_str +=(start+end)
            weight_str +=str(weight)
            if i != (len(cur_node_list)-2):
                path_str+='+'
                weight_str+='+'
        print("the weight sum=", end='')
        print(path_str)
        print("              =",weight_str)
        print("              =",str(sum([d for _,_,d in edge_data])))
class nearestNeightborInsert(nearestNeighbor):
    def __init__(self,G:nx.Graph,seed):
        self.Graph = G
        self.seed = seed
        self.nodes = list(G.nodes())
        self.hamiltion_graph = nx.DiGraph()


    def generate_hamilton(self):
        # ① 构建初始化回路，② 找到回路外距离回路最近的点，进行插入，寻找到最短路径 ③ 重复②直到回路覆盖所有结点。

        np.random.seed(self.seed)
        random_idx = np.random.randint(0, len(self.nodes))
        ran_start_node = self.nodes[random_idx]
        self.hamiltion_graph.add_node(ran_start_node)
        self.cur_node_list=[ran_start_node,ran_start_node]  #回路
        self.cur_path_length = 0


        while len(self.cur_node_list)!=(len(self.nodes)+1):
            print(self.cur_node_list)
            selections ={}
            for i in self.cur_node_list:

                weight_list = self.Graph.edges(i, data='weight')
                sorted_weight_list = sorted(weight_list, key=lambda t: t[2])
                sorted_weight_list = [i for i in sorted_weight_list if i[1] not in self.cur_node_list]
                start,end,weight=sorted_weight_list[0]
                selections[end]=weight
            sorted_selections = sorted(selections.items(),key=lambda x:x[1])

            add_node = sorted_selections[0][0]


            if (len(self.cur_node_list)) == 2:
                edge_weight = self.Graph[self.cur_node_list[0]][add_node]["weight"]

                self.cur_node_list.insert(1, add_node)
                self.cur_path_length = edge_weight * 2
            else:
                path_dict = {}  # 添加的位置和对应的长度
                for i in range(len(self.cur_node_list)-1):

                        rm_edge = self.Graph[self.cur_node_list[i]][self.cur_node_list[i + 1]]["weight"]
                        add_edge1 = self.Graph[self.cur_node_list[i]][add_node]["weight"]
                        add_edge2 = self.Graph[add_node][self.cur_node_list[i + 1]]["weight"]
                        path_dict[i]=self.cur_path_length-rm_edge+add_edge1+add_edge2
                sorted_path_dict = sorted(path_dict.items(),key=lambda x:x[1])
                position,min_path=sorted_path_dict[0][0],sorted_path_dict[0][1]
                self.cur_node_list.insert(position+1,add_node)
                self.cur_path_length = min_path
        print(self.cur_node_list)
        for i in range(len(self.cur_node_list)-1):
            u=self.cur_node_list[i]
            v=self.cur_node_list[i+1]
            self.hamiltion_graph.add_edge(u,v,weight=self.Graph[u][v]['weight'])
        self.print_hamilton_weight_sum()
        return self.hamiltion_graph
def visualize(G:nx.Graph,pos):
    edge_weights = nx.get_edge_attributes(G, 'weight')

    nx.draw_networkx_nodes(G, pos,
                           node_color='#C0D9D9', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, arrows=False, )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.show()

if __name__ == '__main__':
    DATA_SEED=41      #set the random generated data seed
    LAYOUT_SEED = 22  #set the layout seed
    INITNODE_SEED = 43   # 41 A 39 B  31 C 42 D  43 E  # set the random start node

    edgeData=generate_data(seed=DATA_SEED,node_num=6,max_weight=20)
    G = nx.Graph()
    G.add_edges_from(edgeData)
    pos = nx.spring_layout(G,seed=LAYOUT_SEED)
    visualize(G,pos)

    # NN
    # nn = nearestNeighbor(G,seed=INITNODE_SEED)
    # hamiltonGraph = nn.generate_hamilton()
    # visualize(hamiltonGraph,pos)

    # NNI
    nni  =nearestNeightborInsert(G,seed=INITNODE_SEED)
    hamiltonGraph=nni.generate_hamilton()
    visualize(hamiltonGraph, pos)