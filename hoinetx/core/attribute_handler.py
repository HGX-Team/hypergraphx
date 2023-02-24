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
        try:
            return self.id2obj[idx]
        except KeyError:
            raise KeyError("No object with id {}.".format(idx))

    def set_attr(self, idx, attr):
        try:
            self.attr[idx] = attr
        except KeyError:
            raise KeyError("No object with id {}.".format(idx))

    def get_attr(self, obj):
        try:
            return self.attr[self.get_id(obj)]
        except KeyError:
            raise KeyError("No object {}.".format(obj))

    def del_obj(self, obj):
        if obj in self.obj2id:
            idx = self.get_id(obj)
            del self.id2obj[idx]
            del self.obj2id[obj]
            del self.attr[idx]
