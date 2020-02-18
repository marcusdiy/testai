import os;

class MyUtils :
    
    @staticmethod
    def maybeMakeDir(path) :
        if not os.path.exists(path):
            os.makedirs(path)
        return os.path.exists(path);

    @staticmethod
    def print(data) :
        print('');
        print('');
        print('');
        repr(data)
        print('');
        print('');
        print('');