#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:25:13 2019

@author: hamza
"""

import os, math
from shutil import copy

parent="objects"
parent2="objectsSplitted"
folders=os.listdir(parent)
#copyfile(src, dst)
for folder in folders[50:]:
    files=os.listdir(os.path.join(parent,folder))
    seventyP=math.ceil(len(files)*0.7)
    fifteenP=math.ceil(len(files)*0.15)
    train=files[:seventyP]
    val=files[seventyP:seventyP+fifteenP]
    test=files[seventyP+fifteenP:seventyP+2*fifteenP]
    if not os.path.exists(os.path.join(parent2,"train",folder)):
        os.makedirs(os.path.join(parent2,"train",folder))
    if not os.path.exists(os.path.join(parent2,"val",folder)):
        os.makedirs(os.path.join(parent2,"val",folder))
    if not os.path.exists(os.path.join(parent2,"test",folder)):
        os.makedirs(os.path.join(parent2,"test",folder))
    for file in train:
        if not os.path.isfile(os.path.join(parent,folder,file)):
            continue
        sourcePath=os.path.join(parent,folder,file)
        destPath=os.path.join(parent2,"train",folder,file)
        copy(sourcePath,destPath)
    for file in val:
        if not os.path.isfile(os.path.join(parent,folder,file)):
            continue
        sourcePath=os.path.join(parent,folder,file)
        destPath=os.path.join(parent2,"val",folder,file)
        copy(sourcePath,destPath)
    for file in test:
        if not os.path.isfile(os.path.join(parent,folder,file)):
            continue
        sourcePath=os.path.join(parent,folder,file)
        destPath=os.path.join(parent2,"test",folder,file)
        copy(sourcePath,destPath)
    print(folder)
    
