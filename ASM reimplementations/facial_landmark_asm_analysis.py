import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from skimage import io, color, feature
from skimage.filters import sobel
from skimage.filters import sobel_h, sobel_v
import dlib
from PIL import Image
from scipy.spatial import Delaunay

def read_landmarks_from_file(file_path):
    images_landmarks = {}
    images_bounding_boxes = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split()
            image_name = tokens[0]

            # Assuming first four points are bounding box corners
            bounding_box = np.array([float(x) for x in tokens[1:9]]).reshape(-1, 2)
            images_bounding_boxes[image_name] = bounding_box

            # Rest of the points are face landmarks
            landmarks = np.array([float(x) for x in tokens[9:]]).reshape(-1, 2)
            images_landmarks[image_name] = landmarks

    return images_landmarks, images_bounding_boxes


def read_data(image_folder, landmarks_file):
    images = []
    landmarks = []
    bounding_boxes = []

    with open(landmarks_file, 'r') as file:
        for line in file.readlines():
            elements = line.split()
            image_file = os.path.join(image_folder, elements[0])
            image = Image.open(image_file)
            images.append(image)
            landmark_coords = [float(coord) for coord in elements[1:]]
            bounding_box = landmark_coords[:4]
            bounding_boxes.append(bounding_box)
            face_model_landmarks = landmark_coords[4:]
            landmarks.append(face_model_landmarks)
    
    return images, landmarks, bounding_boxes

def display_image_with_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    for idx, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    cv2.namedWindow('Image with Landmarks')
    cv2.imshow('Image with Landmarks', image)

    # Wait for the 'q' key to be pressed to exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
def plot_image_with_landmarks(image, landmarks):
    image_copy = image.copy()
    for idx, point in enumerate(landmarks):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_copy, (x, y), 2, (0, 0, 255), -1)
        #cv2.putText(image_copy, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

###Function for performing Procrustes analysis
def generalized_procrustes_analysis(landmarks_list, max_iterations=1, tolerance=1e-10):
    mean_shape = np.mean(landmarks_list, axis=0)
    for iteration in range(max_iterations):
        aligned_landmarks_list = []
        
        for landmarks in landmarks_list:
            _, aligned_landmarks, _ = procrustes(mean_shape, landmarks)
            aligned_landmarks_list.append(landmarks-aligned_landmarks)
            
        new_mean_shape = np.mean(aligned_landmarks_list, axis=0)
        mean_shape_change = np.linalg.norm(new_mean_shape - mean_shape)

        mean_shape = new_mean_shape

        if mean_shape_change <= tolerance:
            break

    return mean_shape, aligned_landmarks_list

###Fit the mean face model to the input image
def fit_asm_to_new_image(image, mean_face, pca, num_iterations=1000, learning_rate=0.01):
    image_gray = color.rgb2gray(image)
    image_edges = feature.canny(image_gray)

    initial_guess = mean_face.copy()

    for _ in range(num_iterations):
        landmarks_gradient = compute_landmarks_gradient(image_edges, initial_guess)
        initial_guess += learning_rate * landmarks_gradient
        initial_guess = align_shape_to_mean_face(initial_guess, mean_face, pca)

    return initial_guess

###Compute the landmark gradients
def compute_landmarks_gradient(image_edges, landmarks):
    sobel_x = sobel_h(image_edges)
    sobel_y = sobel_v(image_edges)
    gradients = []

    for x, y in landmarks:
        x, y = int(x), int(y)

        gradient_x = -sobel_x[y, x]
        gradient_y = -sobel_y[y, x]

        gradients.append([gradient_x, gradient_y])

    return np.array(gradients)

###Align shape to the mean face
def align_shape_to_mean_face(landmarks, mean_face, pca):
    _, aligned_landmarks, _ = procrustes(mean_face, landmarks)
    pca_aligned_landmarks = pca.transform([aligned_landmarks.reshape(-1)])[0]
    return pca.inverse_transform([pca_aligned_landmarks])[0].reshape(landmarks.shape)

### Dlib's face detector
detector = dlib.get_frontal_face_detector()

def transform_to_bounding_box(mean_shape, bounding_box):
    min_coords = np.min(mean_shape, axis=0)
    max_coords = np.max(mean_shape, axis=0)
    mean_bbox = [min_coords[0], min_coords[1], max_coords[0], max_coords[1]]

    scale_x = (bounding_box[2] - bounding_box[0]) / (mean_bbox[2] - mean_bbox[0])
    scale_y = (bounding_box[3] - bounding_box[1]) / (mean_bbox[3] - mean_bbox[1])

    scale = min(scale_x, scale_y)

    mean_shape_center = np.mean(mean_shape, axis=0)
    bounding_box_center = [(bounding_box[0] + bounding_box[2]) / 2, (bounding_box[1] + bounding_box[3]) / 2]

    translation = bounding_box_center - mean_shape_center

    transformed_shape = mean_shape * scale + translation
    return transformed_shape

def adjust_shape(image, shape):
    # Compute the image gradient
    gradient = sobel(image)

    # For each landmark
    for i in range(shape.shape[0]):
        x, y = int(shape[i, 0]), int(shape[i, 1])

        # Ensure coordinates are within image bounds
        x = max(min(x, image.shape[1] - 1), 0)
        y = max(min(y, image.shape[0] - 1), 0)

        # Get gradient direction
        dx, dy = np.gradient(gradient[max(y-1,0):min(y+2, image.shape[0]), max(x-1,0):min(x+2, image.shape[1])])
       
        # Move landmark in the gradient direction
        shape[i, 0] += dx[1, 1]  # Gradient x at the center of the window
        shape[i, 1] += dy[1, 1]  # Gradient y at the center of the window
        print(shape)
    return shape

def refine_shape(image, initial_shape, mean_shape, pca, num_iterations):
    current_shape = initial_shape.copy()

    for _ in range(num_iterations):
        # Adjust the shape based on the image features
        current_shape = adjust_shape(image, current_shape)

        # Project the shape back to the shape space
        shape_pca = pca.transform([current_shape.reshape(-1)])[0]
        current_shape = pca.inverse_transform([shape_pca])[0].reshape(current_shape.shape)

        # Align the shape with the mean shape
        # _, current_shape, _ = procrustes(mean_shape, current_shape)
        
        input_image_path = "/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM_experiments/FRGC/image/02463d453.jpg"
        image1 = cv2.imread(input_image_path)
        for (x, y) in current_shape:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
    return current_shape


def fit_model_to_image(image_path, mean_shape, pca):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not load image")
        return None

    # Detect faces in the image
    faces = detector(image, 1)

    if len(faces) == 0:
        print("No faces found in the image.")
        return None

    # Use the first detected face
    d = faces[0]
    # Transform the mean shape to fit within the bounding box
    initial_shape = transform_to_bounding_box(mean_shape, [d.left(), d.top(), d.right(), d.bottom()])
    
    for (x, y) in initial_shape:
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
    # Refine the shape to better fit the target face
    # final_shape = refine_shape(image, initial_shape, mean_shape, pca,num_iterations=20)
    
    # input_image_path = "/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM_experiments/FRGC/image/02463d453.jpg"
    # image1 = cv2.imread(input_image_path)
    # for (x, y) in final_shape:
    #     cv2.circle(image1, (int(x), int(y)), 3, (0, 255, 0), -1)
    # cv2.imshow("Image", image1)
    # cv2.waitKey(0)
    # Perform several iterations of landmark adjustments
    for _ in range(50):
        # TODO: adjust the shape based on the image features
        final_shape = adjust_shape(image, initial_shape)
        # pass
    #return initial_shape
    return final_shape



def main():
    landmarks_file_path = 'Data/300W/Train/300W_train.txt'
    images_folder = 'Data/300W/Train'
    images_landmarks, images_bounding_boxes = read_landmarks_from_file(landmarks_file_path)

    ### Display the first image with landmarks
    first_image_name, first_image_landmarks = list(images_landmarks.items())[0]
    first_image_path = os.path.join(images_folder, first_image_name)
    print(first_image_path)
    display_image_with_landmarks(first_image_path, first_image_landmarks)
    
    ##Display first 5 images on the plot to compare 
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for idx, (image_name, landmarks) in enumerate(list(images_landmarks.items())[:5]):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_name}")
            continue

        image_with_landmarks = plot_image_with_landmarks(image, landmarks)
        axes[idx].imshow(image_with_landmarks)
        axes[idx].set_title(image_name)
        axes[idx].axis('off')

    plt.show()
    
    ###Plot all the landmarks as a point cloud.
    all_landmarks = []

    for image_name, landmarks in images_landmarks.items():
        all_landmarks.extend(landmarks)

    all_landmarks = np.array(all_landmarks)

    plt.figure(figsize=(10, 10))
    plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], s=3, c='r')
    plt.title('Point Cloud of All Landmarks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()

    ##Point cloud of only first 10 images
    all_landmarks = []

    for idx, (image_name, landmarks) in enumerate(images_landmarks.items()):
        if idx >=1:
            break
        all_landmarks.extend(landmarks)

    all_landmarks = np.array(all_landmarks)

    plt.figure(figsize=(10, 10))
    plt.scatter(all_landmarks[:, 0], all_landmarks[:, 1], s=5, c='r')
    plt.title('Point Cloud of First 10 Images Landmarks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()
    
    ###Perform PCA on all the landmarks and plot the mean shape
    all_landmarks = []

    for idx, (image_name, landmarks) in enumerate(images_landmarks.items()):
        all_landmarks.append(np.array(landmarks))

    mean_shape, aligned_landmarks_list = generalized_procrustes_analysis(all_landmarks)

    pca = PCA(n_components=2)
    pca.fit(np.array(aligned_landmarks_list).reshape(len(aligned_landmarks_list), -1))

    mean_face = pca.mean_.reshape(mean_shape.shape)

    # Visualize the mean face
    plt.figure(figsize=(10, 10))
    plt.scatter(mean_face[:, 0], mean_face[:, 1], s=30, c='r', marker='o')
    plt.title('Mean Face Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()
    
    # Compute Delaunay triangulation
    tri = Delaunay(mean_face)

    # Visualize the mean face
    plt.figure(figsize=(8, 8))
    plt.scatter(mean_face[:, 0], mean_face[:, 1], s=30, c='r', marker='o')

    # Draw lines between landmarks
    for simplex in tri.simplices:
        plt.plot(mean_face[simplex, 0], mean_face[simplex, 1], 'b-')

    plt.title('Mean Face Model')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()
    
    #Mean face with labelled landmarks 
    # Visualize the mean face
    plt.figure(figsize=(10, 10))
    plt.scatter(mean_face[:, 0], mean_face[:, 1], s=50, c='r', marker='o')
    plt.title('Mean Face Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Annotate each point with its index
    for i, point in enumerate(mean_face):
        plt.annotate(str(i), (point[0], point[1]))

    plt.gca().invert_yaxis()
    # Save the plot
    plt.savefig('mean_face.png')
    plt.show()


    # Plotting the mean shape with connected landmarks
    plt.figure(figsize=(10, 10))
    plt.plot(mean_face[:, 0], mean_face[:, 1], 'ro-', markersize=2)
    plt.title('Mean Face Model with Connected Landmarks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()
    
    # Create a new figure
    plt.figure(figsize=(10, 10))

    # Iterate over all aligned landmarks
    for aligned_landmarks in aligned_landmarks_list:
        # Plot each set of aligned landmarks
        plt.plot(aligned_landmarks[:, 0], aligned_landmarks[:, 1], 'o-', markersize=2)

    # Add some details to the plot
    plt.title('All Aligned Landmarks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y-axis to align with image coordinate system
    plt.show()

    
    ###Calculate covariance matrix and display variations 
    # # Reshape the landmarks data to 2D
    landmarks_data = np.array(aligned_landmarks_list).reshape(len(aligned_landmarks_list), -1)

    # # Subtract the mean from the landmarks data
    landmarks_data = landmarks_data - mean_face.flatten()

    # # Compute the covariance matrix
    cov_matrix = np.cov(landmarks_data, rowvar=False)

    # # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # # Sorting the eigenvectors based on the eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    # # Display the eigenvalues
    plt.figure(figsize=(10, 10))
    plt.plot(eigenvalues)
    plt.title('Eigenvalues of the Covariance Matrix')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.show()

    #Number of standard deviations
    std_dev = 1.5

    # Create a figure
    plt.figure(figsize=(9, 9))

    # # For the first few modes of variation
    for i in range(min(5, eigenvectors.shape[1])):
        # Compute the variation
        variation = mean_face.flatten() + std_dev * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]

        # Reshape the variation to the original shape
        variation = variation.reshape(mean_face.shape)
        plt.gca().invert_yaxis()
        # Plot the variation
        plt.plot(variation[:, 0], variation[:, 1], label=f'Mode {i + 1}')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()
    
    # Calculate and visualize variations in a scatter plot
    # fig, ax = plt.subplots(3, 3, figsize=(20, 20))

    # # Plot the mean face
    # ax[0, 0].scatter(mean_face[:, 0], mean_face[:, 1], s=50, c='r', marker='o')
    # ax[0, 0].set_title('Mean Face')
    # ax[0, 0].invert_yaxis()

    # # Plot the mean face +/- 3 standard deviations along each principal component
    # for i in range(min(4, len(eigenvectors))):
    #     for std_dev in [-3, 3]:
    #         variation = mean_face.flatten() + std_dev * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
    #         variation = variation.reshape(mean_face.shape)

    #         row = (i + 1) // 3
    #         col = (i + 1) % 3
    #         ax[row, col].scatter(variation[:, 0], variation[:, 1], s=50, c='r', marker='o')
    #         ax[row, col].set_title('Variation of Principal Component {}'.format(i + 1))
    #         ax[row, col].invert_yaxis()

    # plt.tight_layout()
    # plt.show()
    
    # Calculate and visualize variations
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))  # Adjust the figsize as needed

    # Plot the mean face
    ax[0, 0].scatter(mean_face[:, 0], mean_face[:, 1], s=10, c='r', marker='o')
    ax[0, 0].set_title('Mean Face')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    # Plot the mean face +/- 3 standard deviations along each principal component
    for i in range(min(4, len(eigenvectors))):
        for std_dev in [-3,3]:
            variation = mean_face.flatten() + std_dev * np.sqrt(eigenvalues[i]) * eigenvectors[:, i]
            variation = variation.reshape(mean_face.shape)

            row = (i + 1) // 3
            col = (i + 1) % 3
            ax[row, col].scatter(variation[:, 0], variation[:, 1], s=10, c=colors[i % len(colors)], marker='o')
            ax[row, col].set_title('Variation of Principal Component {}'.format(i + 1))
            #invert 
            plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    
    # Fit the model to an input image
    input_image_path = "/Users/jagmohanmeher/Documents/NCKU/4th sem/ASM_experiments/300W/Train/image/lfpw/testset/image_0001.jpg"
    fitted_shape = fit_model_to_image(input_image_path, mean_face, pca)
    # # print(fitted_shape)
    if fitted_shape is not None:
        # Visualize the fitting result
        image = cv2.imread(input_image_path)
        for (x, y) in fitted_shape:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    
if __name__ == '__main__':
    main()
