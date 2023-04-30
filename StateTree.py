class Node():
    def __init__(self, data):
        self.data = data
        self.children = []
        self.boxes_avail = None

    def addNode(self, obj):
        self.children.append(obj)

    def set_boxes_avail(self, boxes_avail: int):
        self.boxes_avail = boxes_avail

    def get_children(self):
        return self.children

    def get_boxes_avail(self):
        return self.boxes_avail

    def get_height(self):
        if len(self.children) == 0:
            return 1
        t = max(a.get_height() + 1 for a in self.get_children())
        return t   

    # depth rather than height to exclude the current state/node. value in number of boxes placed after current state/node (depth)
    def get_depth(self):
        t = self.get_height()
        return t - 1