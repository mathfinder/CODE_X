import cv2
import numpy as np
import os
root1 = '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/multi_makeup/results'
root2 = '/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/stargan_multi_makeup/results'
files1 = os.listdir(root1)
files2 = os.listdir(root2)
result = []
for index, name in enumerate(files1):
    if index > 50:
        break
    print index, name
    img1 = cv2.imread(os.path.join(root1, name))
    img2 = cv2.imread(os.path.join(root2, name))
    # print img1.shape
    # print img2.shape

    result.append(img1)
    result.append(img2)
    # line = np.zeros((img.shape[0], 5, 3))
    # for j, vis_dir in enumerate(vis_dirs):
    #     pred = cv2.imread(vis_dir + '/'+ name+ '.png',flags=cv2.IMREAD_UNCHANGED)
    #     if pred.shape[0] != img.shape[0]:
    #         print 'fuck!!'
    #     result.append(pred)
    #     result.append(line)

    #print result_viz.shape

fuse = np.concatenate(tuple(result), 0)
cv2.imwrite('vis/hah.png', fuse)
