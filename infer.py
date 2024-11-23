import subprocess
import sys
import argparse
import os


def install_libraries(libraries):
    """
    Automatically install the specified libraries using pip.

    Args:
        libraries (list): A list of library names to install.
    """
    for library in libraries:
        try:
            # Try importing the library to check if it's already installed
            __import__(library)
            print(f"{library} is already installed.")
        except ImportError:
            # If not installed, use subprocess to install it
            print(f"{library} is not installed. Installing now...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print(f"{library} has been installed successfully.")


# Example usage
if __name__ == "__main__":
    libraries_to_install = [
        "torch",  # PyTorch
        "segmentation_models_pytorch",  # Segmentation Models PyTorch,
        "opencv-python",
        "numpy",
        "albumentations",
        "pillow"
    ]
    install_libraries(libraries_to_install)

    import torch
    import segmentation_models_pytorch as smp
    import cv2
    import numpy as np
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    from PIL import Image

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 3
    ACTIVATION = 'softmax'

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=CLASSES,
        activation=ACTIVATION
    )

    checkpoint = torch.load('../Neoplastic-Segmentation/model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)

    color_dict = {0: (0, 0, 0),
                  1: (0, 0, 255),
                  2: (0, 255, 0)}


    def mask_to_rgb(mask, color_dict):
        output = np.zeros((mask.shape[0], mask.shape[1], 3))

        for k in color_dict.keys():
            output[mask == k] = color_dict[k]

        return np.uint8(output)


    parser = argparse.ArgumentParser(description="Process an image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file.')
    args = parser.parse_args()

    val_transformation = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


    # Parse arguments
    def get_image_path(image_path):
        """
        Validates and returns the path of the input image file.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The validated absolute path of the image file.
        """
        try:
            # Check if the file exists
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"The file {image_path} does not exist.")

            # Get the absolute path
            abs_path = os.path.abspath(image_path)
            print(f"Image path validated: {abs_path}")
            return abs_path
        except FileNotFoundError as e:
            print(e)
            return None


    img_path = get_image_path(args.image_path)
    print("Image Path:", img_path)
    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_w = ori_img.shape[0]
    ori_h = ori_img.shape[1]
    img = cv2.resize(ori_img, (256, 256))
    transformed = val_transformation(image=img)
    input_img = transformed["image"]
    input_img = input_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output_mask = model.forward(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    mask = cv2.resize(output_mask, (ori_h, ori_w))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(mask_rgb)
    # Show the image using PIL
    pil_image.show()


    def save_image_to_current_directory(rgb_array, output_image_name):
        """
        Saves an image to the current directory given a NumPy RGB array.

        Args:
            rgb_array (numpy.ndarray): The RGB image as a NumPy array.
            output_image_name (str): The name for the saved image (e.g., "output_image.jpeg").
        """
        try:
            # Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(rgb_array)

            # Get the current working directory
            current_directory = os.getcwd()

            # Create the full path for the output image
            output_path = os.path.join(current_directory, output_image_name)

            # Save the image to the current directory
            pil_image.save(output_path)
            print(f"Image saved successfully to: {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Name of the output image to be saved
    output_image_name = f"{img_path}_prediction.jpeg"

    # Save the image to the current directory
    save_image_to_current_directory(mask_rgb, output_image_name)
