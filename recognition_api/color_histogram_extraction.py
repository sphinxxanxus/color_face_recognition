from cv2 import split, calcHist, imread
from numpy import argmax as argmax
from os import listdir
from time import sleep


def maxs_pixel_rgb(image_read: str)->str:
    global green, blue, red
    image_split = split(image_read)
    colors: tuple = ('b', 'g', 'r')
    counter: int = 0
    for (color_pixel, i) in zip(image_split, colors):
        counter += 1
        hist = calcHist([color_pixel], [0], None, [256], [0, 256])
        max_pixel_elem = argmax(hist)  # Prend le peack des pixels
        if counter == 1:
            blue = str(max_pixel_elem)
        elif counter == 2:
            green = str(max_pixel_elem)
        elif counter == 3:
            red = str(max_pixel_elem)
    return red + "," + green + "," + blue


def color_histogram_image(image_read: str):
    with open('test.data', 'w', encoding='utf8') as file_open:
        file_open.write(maxs_pixel_rgb(image_read))


def color_histogram_training_image(image_name: str):
    """Function d'entrainement de l'algorithme"""
    data_name = ''
    color = {1: "red", 2: "white", 3: "yellow", 4: "black",
             5: "violet", 6: "blue", 7: "green", 8: "orange"}
    for (k, value) in color.items():
        if value in image_name:
            data_name = value
            break
    image_read = imread(image_name)
    with open('training.data', 'a', encoding='utf8') as file_append:
        file_append.write(maxs_pixel_rgb(image_read) + ',' + data_name + '\n')


def training() -> None:
    """fonction qui entraine le modele de classification"""
    folder_path: list[str] = listdir('./training_dataset')
    for folder_name in folder_path:
        for file_name in listdir('./training_dataset/'+folder_name):
            color_histogram_training_image('./training_dataset/'
                                           + folder_name + '/'
                                           + file_name)
        print("\033[1;34m Entraînement à la "
              "couleur " + folder_name + "...\033[1;m"
              + "\033[1;32m done!!!\033[1;m", flush=True)
        sleep(2)
