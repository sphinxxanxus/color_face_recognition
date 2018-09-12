import math
from sklearn import neighbors
from os import path, listdir
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from time import sleep

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def training(train_dir: str, model_save_path: str = None,
             n_neighbors: int = 3, knn_algo: str = 'ball_tree',
             verbose=False):
    x: list = []
    y: list = []
    for class_dir in listdir(train_dir):
        if not path.isdir(path.join(train_dir, class_dir)):
            continue

        for image_path in image_files_in_folder(path.join(train_dir,
                                                          class_dir)):
            image = face_recognition.load_image_file(image_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people)
                # in a training image, skip the image.
                if verbose:
                    print("\033[1;31mImage {} ne peut être utilisée pour"
                          " l'entraînement: {}\033[1;m".format(
                            image_path, "Didn't find a face"
                            if len(face_bounding_boxes) < 1
                            else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                x.append(face_recognition.face_encodings(
                    image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

        print("\033[1;34m Entraînement à la reconnaissance de "
              + class_dir + "...\033[1;m"
              + "\033[1;32m done!!!\033[1;m", flush=True)
        sleep(2)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 algorithm=knn_algo,
                                                 weights='distance')
        knn_clf.fit(x, y)

        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)


def predict(x_img_path: str, knn_clf=None, model_path: str=None,
            distance_threshold=0.6):
    if not path.isfile(x_img_path) or path.splitext(x_img_path)[1][
                                         1:] not in ALLOWED_EXTENSIONS:
        raise Exception("\033[1;31m Invalid image path: "
                        "{}\033[1;m".format(x_img_path))

    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

        # Load image file and find face locations
    x_img = face_recognition.load_image_file(x_img_path)
    x_face_locations = face_recognition.face_locations(x_img)

    # If no faces are found in the image, return an empty result.
    if len(x_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        x_img, known_face_locations=x_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in
                   range(len(x_face_locations))]

    # Predict classes and remove classifications
    # that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), x_face_locations,
                are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width = 20
        text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)),
                       fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5),
                  name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


def main(image_file_path):
    # Find all people in the image using a trained classifier model
    # Note: You can pass in either
    # a classifier file name or a classifier model instance
    predictions = predict(image_file_path, model_path="training_face.clf")

    # Print results on the console
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

    # Display results overlaid on an image
    show_prediction_labels_on_image(image_file_path, predictions)
