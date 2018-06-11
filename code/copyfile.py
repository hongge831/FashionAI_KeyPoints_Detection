from tqdm import tqdm
import os


def coverFiles(sourceDir, targetDir):
    for file in os.listdir(sourceDir):
        sourceFile = os.path.join(sourceDir, file)
        targetFile = os.path.join(targetDir, file)
        # cover the files
        if os.path.isfile(sourceFile):
            open(targetFile, "wb").write(open(sourceFile, "rb").read())


fileList1 = os.listdir('/home/tanghm/Documents/YFF/project/data/evaluation/Images/')
# fileList2 = os.listdir('E:/alikeypoint/YFF/project/data/train/Images')
srcfile = '/home/tanghm/Documents/YFF/project/data/evaluation/Images/'
desfile = '/home/tanghm/Documents/YFF/project/data/train/Images/'
for fileName in tqdm(fileList1):
    coverFiles(srcfile+fileName,desfile+fileName)