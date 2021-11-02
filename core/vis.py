from graphviz import Digraph
import imageio
import cv2
import numpy as np


def show_tree(root, file_path='test.gv'):
    graph = Digraph()

    # bfs
    node_stack = [root]
    edges = []
    while len(node_stack) > 0:
        node = node_stack.pop()
        assert node.tag is not None

        for action, child in node.children.items():
            if child.visit_count > 0 and child.tag is not None:
                edges.append((str(node.tag), str(child.tag)))
                node_stack.append(child)

    graph.edges(edges)
    graph.render(file_path, view=False)


class DrawNode(object):
    _tree_num = 0

    def __init__(self, action):
        self.action = action
        self.parent = None
        self.children = {}
        self.index = DrawNode._tree_num
        DrawNode._tree_num += 1

    def add_child(self, action):
        if action not in self.children.keys():
            node = DrawNode(action)
            node.parent = self
            self.children[action] = node

    def add_traj(self, traj):
        node = self
        for action in traj:
            node.add_child(action)
            node = node.children[action]

    @staticmethod
    def clear():
        DrawNode._tree_num = 0


class DrawTree(object):
    def __init__(self, root):
        self.root = root

        self.dot = Digraph(format='png')
        self.build()

    def add_node(self, index, action):
        if index == '0':
            tag = 'Root#0'
        else:
            tag = 'Action[{}]#{}'.format(action, index)

        self.dot.node(index, tag)

    def build(self):
        self.dot.clear()
        node_stack = [self.root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            index = node.index
            action = node.action
            self.add_node(str(index), str(action))

            for child in node.children.values():
                self.dot.edge(str(index), str(child.index))
                node_stack.append(child)
        self.show()

    def show(self):
        num = DrawNode._tree_num
        path = 'temp/tree_{}'.format(num)
        self.dot.render(path, view=False)

    def make_video(self):
        w = imageio.get_writer('tree_video.mp4')
        for i in range(1, DrawNode._tree_num + 1):
            img = cv2.imread('temp/tree_{}.png'.format(i))
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA).astype(np.uint8)

            for _ in range(10):
                w.append_data(img)
        w.close()


if __name__=='__main__':

    trajectories = [
        [2],
        [4],
        [2, 3]
    ]

    DrawNode.clear()
    root = DrawNode(0)

    draw_tree = DrawTree(root)
    for traj in trajectories:
        root.add_traj(traj)
        draw_tree.build()
    draw_tree.make_video()
