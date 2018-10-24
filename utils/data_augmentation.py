import os
import imageio
import Augmentor
import numpy as np

def save_label_to_file(data, label, image_size,
                       max_num=None, path="augment/input/"):
    """Save the dictionary images of a label to file.
    Parameters
    ----------
    data : dictionary
        The dataset keyed with labels.
    label : key
        The key of the specific dictionary part.
    image_size : 2D tupple
        Size of the original images eg. (28,28)
    max_num : int
        maximum number of images
    path : str
        Path to save
    Returns
    -------
    int
        The number of saved images"""
    if not os.path.exists(path):
        os.makedirs(path)
    if(max_num==None):
        max_num = len(data[label])
    sample_num = min(len(data[label]),max_num)
    for i in range(sample_num):
        image = np.resize(data[label][i],image_size)
        image = np.array(image).astype("uint8")
        imageio.imwrite(path+"test"+str(i)+".PNG", image)
    return sample_num

def generate_augmented_data(sample_num, augm_type,
                            input_path="augment/input/",
                            output_path="../output/"):
    """Generates images to files
    Parameters
    ----------
    sample_num : int
        Number of generated images from every image in the path
    augm_type : str
        Type of the augmentation. Implemented: rotate,
        shear, distortion, all
    input_path : str
        Path to the imput files
    output_part : str
        Path to the output files, add type as a subfolder"""
    pipe = Augmentor.Pipeline(input_path,
                              output_path+augm_type+"/",
                              save_format="PNG")
    if(augm_type=="rotate" or augm_type=="all"):
        pipe.rotate(probability=1,
                    max_left_rotation=7,
                    max_right_rotation=7)
    if(augm_type=="shear" or augm_type=="all"):
        pipe.shear(probability=1,
                   max_shear_left=15,
                   max_shear_right=15)
    if (augm_type=="distortion" or augm_type=="all"):
        pipe.random_distortion(probability=1,
                               grid_height=5,
                               grid_width=5,
                               magnitude=2)
    pipe.sample(sample_num)
