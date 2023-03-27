class MetaHandler:
    def __init__(self):
        self.id = 0
        self.id2obj = {}
        self.obj2id = {}
        self.attr = {}

    def add_obj(self, obj, obj_type=None):
        if obj in self.obj2id:
            raise KeyError("Object {} already exists.".format(obj))
        self.id2obj[self.id] = obj
        self.obj2id[obj] = self.id
        self.attr[self.id] = {'type': obj_type, 'name': obj}
        self.id += 1
        return self.id - 1

    def get_id(self, obj):
        try:
            return self.obj2id[obj]
        except KeyError:
            raise KeyError("No object {}.".format(obj))

    def get_obj(self, idx):
        try:
            return self.id2obj[idx]
        except KeyError:
            raise KeyError("No object with id {}.".format(idx))

    def set_attr(self, obj, attr):
        idx = self.get_id(obj)
        try:
            self.attr[idx] = attr
        except KeyError:
            raise KeyError("No object with id {}.".format(idx))

    def get_attr(self, obj):
        try:
            return self.attr[self.get_id(obj)]
        except KeyError:
            raise KeyError("No object {}.".format(obj))

    def add_attr(self, obj, attr, value):
        idx = self.get_id(obj)
        self.attr[idx][attr] = value

    def remove_attr(self, obj, attr):
        idx = self.get_id(obj)
        del self.attr[idx][attr]

    def remove_obj(self, obj):
        if obj in self.obj2id:
            idx = self.get_id(obj)
            del self.id2obj[idx]
            del self.obj2id[obj]
            del self.attr[idx]
