import cv2

# Read the binary image
binary_image = cv2.imread('1341.png', cv2.IMREAD_GRAYSCALE)

# Invert the binary image
inverted_image = cv2.bitwise_not(binary_image)

# Convert the inverted image to RGB
rgb_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2RGB)

# Display the original and inverted images
cv2.imshow('Original Binary Image', binary_image)
cv2.imshow('Inverted RGB Image', rgb_image)
cv2.waitKey(0)
cv2.destroyAllWindows()