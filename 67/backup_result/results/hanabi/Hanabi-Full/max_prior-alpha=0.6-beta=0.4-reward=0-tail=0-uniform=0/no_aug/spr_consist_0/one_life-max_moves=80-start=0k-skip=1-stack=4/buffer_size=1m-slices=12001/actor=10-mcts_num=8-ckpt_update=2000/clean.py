import os
for root, dirs, files in os.walk("."):
        for file in files:
                    if file.endswith("train.log"):
                                     valid=False
                                     with open(file,'r') as reader:
                                         for epoch in ['1000','2000','10000','5000']:
                                             if epoch in reader:
                                                 valid=True

                                     print(os.path.join(root, file), valid)
                                     if not valid:
                                         os.rmdir(root.parent.absolute())

