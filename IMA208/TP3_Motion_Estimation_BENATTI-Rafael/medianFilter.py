import numpy as np

def medianFilter(img, k=3):
    # Applies a median filter to an input image using a square window of size k x k
    
    # Compute the padding size for the input image
    pad_size = k // 2
    
    # Pad the input image with zeros
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    
    # Initialize the output filtered image
    filtered_img = np.zeros_like(img)
    
    # Iterate over each pixel in the output image
    for i in range(pad_size, img.shape[0] + pad_size):
        for j in range(pad_size, img.shape[1] + pad_size):
            # Extract the subimage corresponding to the current window
            window = padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            
            # Compute the median of the window
            median_val = np.median(window)
            
            # Set the corresponding pixel in the filtered image to the median value
            filtered_img[i-pad_size, j-pad_size] = median_val
            
    return filtered_img
