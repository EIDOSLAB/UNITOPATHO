import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor


size = [224, 224]
batch_size = 64

def UNITOPatho(resolution = 800):

    if resolution == 800:
        #for 800 micron resolution images
        image_size = 1812
        ds_file = 'unitopath-public/800/unitopath-public-800.yml'
    elif resolution == 7000:
        #for 7000 micron resolution images
        image_size = 15855
        ds_file = 'unitopath-public/7000/unitopath-public-7000.yml'
    else:
        print(f'Resolution must be 800 or 7000, got {resolution}')
        exit(1)

    base_augmention = []
    while image_size//2 > size[0]:
        base_augmention.append(ecvl.AugResizeDim([image_size//2,image_size//2]))
        image_size = image_size//2
    base_augmention.append(ecvl.AugResizeDim(size))

    #Augmentation examples
    training_augs = ecvl.SequentialAugmentationContainer( base_augmention + [
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180])
    ])

    #Augmentationd for training, validation and test sets
    dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, None, None]
    )

    return ecvl.DLDataset(ds_file, batch_size, dataset_augs)

dataset = UNITOPatho(resolution = 800)

#set dataset to the test-set
dataset.SetSplit(ecvl.SplitType.test)

#etc ...
pass

exit()

