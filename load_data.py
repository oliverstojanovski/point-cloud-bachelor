import cv2
import pathlib

TARGET_FILE_PATH = pathlib.Path('pointclouds/3DFlasche.ply')
SOURCE_FILE_PATH = pathlib.Path('pointclouds/Glasflaschen_1.ply')

'''
This function loads grayscale images from specified paths.

The function reads the grayscale images from file paths that are defined as global variables "TARGET_FILE_PATH" and "SOURCE_FILE_PATH", 
which contain the file paths of the target and source images.

The function returns a tuple of two grayscale images, "img" is the grayscale target image and "template" is the grayscale source image


'''


def load_data():
    img = cv2.imread(str(TARGET_FILE_PATH), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    template = cv2.imread(str(SOURCE_FILE_PATH), cv2.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    return img, template
