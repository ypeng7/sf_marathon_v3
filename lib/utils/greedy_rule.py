import copy
import numpy as np
from utils.a_star import AStar, Array2D, Point
import random

class GreedyBenchmark(object):
    def __init__(self, state_inp):
        self.field_size = state_inp.shape[0]
        #创建一个10*10的地图
        self.map2d=Array2D(self.field_size,self.field_size)

        for i in range(self.field_size):
            for j in range(self.field_size):
                if state_inp[i][j][0] == 1:
                    self.map2d[i][j] = 1

        #显示地图当前样子
        # self.map2d.showArray2D()


    def proceed_step(self, pkg_values, self_coord):
        pkgs = np.argwhere(np.greater(pkg_values, np.zeros_like(pkg_values)))

        def get_path(p1x, p1y, p2x, p2y):
            p1 = Point(p1x, p1y)
            p2 = Point(p2x, p2y)
            astar = AStar(self.map2d, p1, p2)
            pathList = astar.start()
            return pathList, len(pathList)

        old_distance_reward = [(pkg_values[x[0], x[1]], get_path(self_coord[0], self_coord[1], x[0], x[1]), x) for x in list(pkgs)]
        old_distance_reward = list(filter(lambda x: x[0] > 0, old_distance_reward))
        values = [(x[0]/x[1][1], x[1][0], x[2]) for x in old_distance_reward]


        top_value_dest_li = list(sorted(values, key=lambda x: x[0], reverse=True))
        if len(top_value_dest_li) == 0:
            movement = random.choice(["U", "D", "L", "R"])
            pathList = None
        else:
            #创建AStar对象,并设置起点为0,0终点为9,0
            top_value_dest = top_value_dest_li[0]
            pathList = top_value_dest[1]
        #遍历路径点,在map2d上以'8'显示
        # for point in pathList:
        #     self.map2d[point.x][point.y]=8
        #     # print(point)
        # print("----------------------")
        #再次显示地图
        # self.map2d.showArray2D()
        if pathList is None:
            movement = random.choice(["U", "D", "L", "R"])
        elif len(pathList) > 0:

            next_coord = pathList[0]
            x0_displacement = next_coord.x - self_coord[0]
            x1_displacement = next_coord.y - self_coord[1]

            movement = ""
            if x0_displacement == 1:
                movement = "D"
            elif x0_displacement == -1:
                movement = "U"
            elif x1_displacement == 1:
                movement = "R"
            elif x1_displacement == -1:
                movement = "L"
            else:
                print("!!!!something wrong with direction!!!")
                movement = random.choice(["U", "D", "L", "R"])
        else:
            movement = random.choice(["U", "D", "L", "R"])

        return movement

