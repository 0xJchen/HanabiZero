import os
def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass
def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))
cur=os.getcwd()
remove_empty_dirs(cur)
