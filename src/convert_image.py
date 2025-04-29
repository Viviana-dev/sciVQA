from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# Load your image
img_path = "data/images/train/images_train/2009.12539v1-Figure3-1.png"
image = Image.open(img_path).convert("RGB")  # ensure RGB format

# Show the image
plt.imshow(image)
plt.axis("off")
plt.show()

# Convert the image to a PyTorch tensor
transform = transforms.ToTensor()
tensor_image = transform(image)

# Check the shape of the tensor
print("Tensor shape:", tensor_image.shape)  # Should be [3, H, W]

# Print pixel values (just a small patch)
print("Top-left corner values (channel 0):")
print(tensor_image[0, :5, :5])  # Print top-left 5x5 block of the red channel
