class licence():
    def __init__(self):
        self.keys = []

    def get_licence_keys(self):
        with open('./conf/licence.txt', 'r') as f:
            self.keys.clear()
            keys = f.readlines()
            for key in keys:
                if key != "":
                    self.keys.append(key.replace("\n",""))

    def check_licence_key(self, key):
        self.get_licence_keys()
        return True if key in self.keys else False
