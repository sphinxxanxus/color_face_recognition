#!/usr/bin/python3

import argparse
from cv2 import imread, putText, FONT_HERSHEY_PLAIN, \
    namedWindow, imshow, waitKey, WINDOW_NORMAL, VideoCapture,\
    destroyAllWindows, resizeWindow
from os import path, access, R_OK, remove

from recognition_api import color_histogram_extraction as color_extraction
from recognition_api import knn_classifier as knn
from recognition_api import knn_face_recognition as knn_face


def cmd_parse_line():
    parser = argparse.ArgumentParser(prog='recognition',
                                     description='Détection de couleur')
    parser.add_argument('-i', '--image', help='Activation du mode image',
                        action='store_true')
    parser.add_argument('-w', '--webcam', help='Activation du mode caméra',
                        action='store_true')
    parser.add_argument('-p', '--path', help='Dossier contenant l\'image',
                        type=str)
    parser.add_argument('-t', '--training', help='Entraîne le modèle',
                        action='store_true')
    parser.add_argument('-k', '--k', help='Nombre de voisin', type=int,
                        default=3)
    parser.add_argument('-c', '--camera', help='Le numéro de caméra.'
                                               ' 0 pour la  webcam du PC',
                        default=0, type=int)
    parser.add_argument('-f', '--face', help='face recognition image',
                        action='store_true')
    return parser.parse_args()


def is_already_training() -> bool:
    """Vérifie si l'algorithme est déjà entrainé ou pas"""
    paths: str = 'training.data'
    if path.isfile(paths) and access(paths, R_OK):
        return True
    return False


if __name__ == '__main__':
    args = cmd_parse_line()  # Récupération des arguments
    if args.image and args.path and args.path != '':
        if is_already_training():
            print("\033[1;32m Je me suis déjà entraîné ^_^\033[1;m \n"
                  "\033[1;34m Classification de l'image en cours ... \033[1;m")
            try:
                image = imread(args.path)
                color_extraction.color_histogram_image(image)
                prediction: str = knn.main('training.data', 'test.data', args.k)
                putText(image, 'Prediction: ' + prediction, (20, 55),
                        FONT_HERSHEY_PLAIN, 3, 100, )
                ## Affichage
                namedWindow('Image Color Classifier', WINDOW_NORMAL)
                resizeWindow('Image Color Classifier', 600, 300)
                imshow('Image Color Classifier', image)
                waitKey(0)
            except FileExistsError:
                print("\033[1;31m Le fichier n'existe pas\033[1;m")
            except FileNotFoundError:
                print("\033[1;31m Aucun de ce type trouver\033[1;m")
        else:
            print("\033[1;31m Je ne suis pas encore entrainé. ^-^\033[1;m")
            print("\033[1;32m Entraînez moi svp!!! \033[1;m")
    elif args.webcam:
        print("\033[1;34m Détection de couleur par caméra \033[1;m")
        try:
            cap = VideoCapture(args.camera)
            prediction: str = ''
            while True:
                (ret, frame) = cap.read()
                putText(frame, 'Prediction: ' + prediction, (15, 45),
                        FONT_HERSHEY_PLAIN, 3, 100,)

                # Display the resulting frame
                namedWindow('Camera Color Classifier', WINDOW_NORMAL)
                resizeWindow('Image Color Classifier', 600, 300)
                imshow('Camera Color Classifier', frame)

                color_extraction.color_histogram_image(frame)

                prediction = knn.main('training.data', 'test.data', args.k)
                if waitKey(1) & 0xFF == ord('q'):
                    break
            # When everything done, release the capture
            cap.release()
            destroyAllWindows()
        except ConnectionError:
            print("\033[1;31m Vérifier le numéro de votre périphérique caméra")
    elif args.training:
        print("\033[1;34m Entraînement en cours ... \033[1;m")
        try:
            remove('test.data')
            remove('training.data')
            remove('training_face.clf')
        except FileNotFoundError:
            pass
        except FileExistsError:
            pass
        color_extraction.training()  # Entraînement
        knn_face.training('training_face_dataset', 'training_face.clf', args.k)
    elif args.face:
        knn_face.main(args.path)