import os
from pathlib import Path
import shutil
for root, dirs, files in os.walk("."):
        #print(root,dirs,files)
        for file in files:
                    if file.endswith("train.log"):
                                     valid=False
                                     print("detecting file", os.path.join(root,file))
                                     with open(os.path.join(root,file),'r') as reader:
                                         for epoch in ['1000','2000','10000','5000']:
                                             if epoch in reader:
                                                 valid=True

                                     print(os.path.join(root, file), valid)
                                     if not valid:
                        #                 cur_file=os.path.join(root,file)
                         #                prnt=cur_file.parent.absolute()
                                         shutil.rmtree(Path(root).parent.absolute())

