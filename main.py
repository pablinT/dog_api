import cv2

# Load the Haar Cascade for dog detection (you need to download or provide the XML file)
# You can find pre-trained cascade classifiers for various objects, including dogs, online.
dog_cascade = cv2.CascadeClassifier('dog_face.xml')

# Function to detect if an image contains a dog
def has_dog(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform the dog detection
    dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If dogs are found, return True; otherwise, return False
    return len(dogs) > 0

# Example usage
image_path = '/home/developer/Documentos/appart_rsch/detector_lara/positive_images/4.jpg'
if has_dog(image_path):
    print('This image contains a dog.')
else:
    print('No dog found in the image.')
