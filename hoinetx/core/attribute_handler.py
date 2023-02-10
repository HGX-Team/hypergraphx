class AttributeHandler:
    def __init__(self):
        self.id = 0
        self.id2obj = {}
        self.obj2id = {}
        self.attr = {}

    def get_id(self, obj):
        if obj in self.obj2id:
            return self.obj2id[obj]
        else:
            self.id2obj[self.id] = obj
            self.obj2id[obj] = self.id
            self.id += 1
            return self.id - 1

    def get_obj(self, idx):
        return self.id2obj[idx]

    def set_attr(self, idx, attr):
        self.attr[idx] = attr

    def get_attr(self, obj):
        return self.attr[self.get_id(obj)]
