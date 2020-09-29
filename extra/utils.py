from os import listdir
from os.path import isdir, join
from sklearn.utils import check_random_state
from matplotlib.image import pil_to_array
from PIL import Image
import numpy as np

#这些数据是按角度信息排过序的，不用shuffle
def load_img(container_path,type1=None,num_classes=10,categories=None, shuffle=False, random_state=0, resize=(128,128)):
    target = []
    target_names = []
    filenames = []
    data = []

    folders = [f for f in sorted(listdir(container_path))
               if isdir(join(container_path, f))]

    if categories is not None:
        folders = [f for f in folders if f in categories]
    folders=folders[:num_classes]
    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        if type1=='train':
            txt_path='datasets/class10/txt'
            txt_p=txt_path+'/'+folder+'.txt'
            f=open(txt_p)
            txt_all=f.readlines()
            documents=[join(folder_path, d.split('.')[0]+'.jpeg') for d in txt_all]

        target.extend(len(documents) * [label])
        filenames.extend(documents)

    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    for file in filenames:
        im = Image.open(file)
        if resize is not None:
            im = im.resize(resize, Image.NEAREST)
        data.append(pil_to_array(im)/255.0)

    return {'target': target, 'data': np.array(data), 'target_names': target_names}