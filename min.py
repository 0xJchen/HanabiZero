import ray
import time
@ray.remote
class foo(object):
    def __init__(self,n):
        self.n=n
    def f(self):
        print("foo sleeping!")
        time.sleep(self.n)
        print("foo wakening!")
        return n
ray.init()
obj=foo.remote(200)
num=obj.f.remote()
ray.wait([num])
print("jump!")
