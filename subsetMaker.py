import os
 
depthFakesPath = './dataset/Faceforensics/Depth-Faces/Fake/c40/Deepfakes'
depthRealPath = './dataset/Faceforensics/Depth-Faces/Real/c40/youtube'

rgbFakesPath = './dataset/Faceforensics/RGB-Faces/Fake/c40/Deepfakes'
rgbRealPath = './dataset/Faceforensics/RGB-Faces/Real/c40/youtube'

depthFakesPathSub = './subset/Faceforensics/Depth-Faces/Fake/c40/Deepfakes'
depthRealPathSub = './subset/Faceforensics/Depth-Faces/Real/c40/youtube'

rgbFakesPathSub = './subset/Faceforensics/RGB-Faces/Fake/c40/Deepfakes'
rgbRealPathSub = './subset/Faceforensics/RGB-Faces/Real/c40/youtube'

depthFakes = []
depthReal = []

rgbFakes = []
rgbReal = []


for it in os.scandir(depthFakesPath):
    if it.is_dir():
        depthFakes.append(it.name)

for it in os.scandir(depthRealPath):
    if it.is_dir():
        depthReal.append(it.name)

for it in os.scandir(rgbFakesPath):
    if it.is_dir():
        rgbFakes.append(it.name)

for it in os.scandir(rgbRealPath):
    if it.is_dir():
        rgbReal.append(it.name)

count = 0 
for elem in depthFakes[:100]:
    dirListingDepth = os.listdir(depthFakesPath+'/'+elem)
    dirListingRGB = os.listdir(rgbFakesPath+'/'+elem)
    if len(dirListingDepth) == len(dirListingRGB):
        count += len(dirListingDepth)
        os.system('cp -r '+depthFakesPath+'/'+elem+' '+depthFakesPathSub+'/'+elem)
        os.system('cp -r '+rgbFakesPath+'/'+elem+' '+rgbFakesPathSub+'/'+elem)


print(count)

count = 0 
for elem in depthReal[:100]:
    dirListingDepth = os.listdir(depthRealPath+'/'+elem)
    dirListingRGB = os.listdir(rgbRealPath+'/'+elem)
    if len(dirListingDepth) == len(dirListingRGB):
        count += len(dirListingDepth)        
        os.system('cp -r '+depthRealPath+'/'+elem+' '+depthRealPathSub+'/'+elem)
        os.system('cp -r '+rgbRealPath+'/'+elem+' '+rgbRealPathSub+'/'+elem)

print(count)
