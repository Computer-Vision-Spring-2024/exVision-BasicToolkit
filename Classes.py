import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors


class NoiseAdder:
    def __init__(self) -> None:
        self.lookup = {
            "uniform": self.uniform,
            "gaussian": self.gaussian,
            "salt & pepper": self.spices
        }
        self.noise_types = self.lookup.keys()

    def apply_noise(self, image, noise_type: str, strength):
        if noise_type.lower() not in self.noise_types:
            raise ValueError(f"Invalid noise type. Allowed types: {', '.join(self.noise_types)}")
        
        return self.lookup[noise_type.lower()](image, strength)

    def uniform(self, image, strength):
        upper_limit = self.map_value(strength, 0, 100, 0, 255) # map up
        noise = np.random.uniform(0, upper_limit, size=image.shape) 
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def gaussian(self, image, strength):
        std = self.map_value(strength, 0, 100, 0, 100) # map up 
        noise = np.random.normal(0, std, size=image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def spices(self, image, strength):
        amount = self.map_value(strength, 0, 100, 0, 1) # map down 
        noisy_image = np.copy(image)
        num_salt = np.ceil(amount * image.size * 0.5)
        salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1]] = 255

        num_pepper = np.ceil(amount * image.size * 0.5)
        pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_image.astype(np.uint8)
    
    def map_value(self, value, lower_range, upper_range, lower_range_new, upper_range_new):

        mapped_value = ((value - lower_range) * (upper_range_new - lower_range_new) /
                        (upper_range - lower_range)) + lower_range_new
        return mapped_value

#------------------------------------------------------------- Image Filter ---------------------------------------------------------------


class ImageFilter:
    def __init__(self) -> None:
        self.lookup = {
            "average": self.average,
            "gaussian": self.fuzzy,
            "median": self.median,
            "bilateral": self.bilateral
        }
        self.filter_types = self.lookup.keys()

    
    def apply_filter(self, image, filter_type: str, filter_size, **kwargs):
        if filter_type.lower() not in self.filter_types:
            raise ValueError(f"Invalid filter type. Allowed types: {', '.join(self.filter_types)}")
        
        if filter_size == 0:
            raise ValueError(f"Invalid kernel size")

        return self.lookup[filter_type.lower()](image, filter_size, **kwargs)
    
    def padding_image(self, image, width, height, pad_size):
        padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size)) 
        padded_image[pad_size:-pad_size, pad_size:-pad_size] = image 
        return padded_image
    
    def get_gaussian_mask(self,filter_size, sigma):
        if sigma != 0: 
            kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (filter_size-1)/2)**2 + (y - (filter_size-1)/2)**2) / (2*sigma**2)), (filter_size, filter_size))
            return kernel
        else:
            raise ValueError("Sigma value can not be zero")

    
    def convolve_2d(self, image, kernel, mutlipy = True):
        image_height, image_width = image.shape
        kernel_size = kernel.shape[0]

        pad_size = kernel_size // 2

        if pad_size == 0:
            padded_image = image
            normalize_value = 2
        else:
            # padding the image to include edges 
            normalize_value = kernel_size * kernel_size 
            padded_image = self.padding_image(image, image_width, image_height, pad_size)
            
        output_image = np.zeros_like(image)
        for i in range(image_height):
            for j in range(image_width):
                neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size] # slice out the region 
                # optimization trick  
                if mutlipy:
                    output_image[i, j] = np.sum(neighborhood * kernel)
                else:
                    output_image[i, j] = np.sum(neighborhood) * (1/normalize_value)
        return np.clip(output_image,0,255)
    

    
    def median_2d(self, image, filter_size):
        image_height, image_width = image.shape
        pad_size = filter_size // 2
        
        padded_image = self.padding_image(image, image_width, image_height, pad_size)
        
        filtered_image = np.zeros_like(image)
        
        for i in range(image_height):
            for j in range(image_width):
                roi = padded_image[i:i+filter_size, j:j+filter_size]
                filtered_image[i, j] = np.median(roi)
        return filtered_image
    
      
    def average(self, image, filter_size, **kwargs):
        kernel = np.ones((filter_size, filter_size)) / (filter_size * filter_size)
        if image.colored:
            num_channels = image.manipulated_img.shape[2]
            filtered_image = np.zeros_like(image.manipulated_img)
            for ch in range(num_channels):
                filtered_image[:,:,ch] = self.convolve_2d(image.manipulated_img[:,:,ch], kernel, False)
        else:
            filtered_image = self.convolve_2d(image.manipulated_img, kernel, False)  
        return filtered_image


    def fuzzy(self, image, filter_size, **kwargs):
        sigma = kwargs.get("sigma", 1.0)
        kernel = self.get_gaussian_mask(filter_size,sigma)
        kernel /= np.sum(kernel)
        if image.colored:
            num_channels = image.manipulated_img.shape[2]
            filtered_image = np.zeros_like(image.manipulated_img)
            for ch in range(num_channels):
                filtered_image[:,:,ch] = self.convolve_2d(image.manipulated_img[:,:,ch], kernel, True)       
        else:
            filtered_image = self.convolve_2d(image.manipulated_img, kernel, True)  
        return filtered_image
  
    def median(self, image, filter_size, **kwargs):
        if image.colored:
            num_channels = image.manipulated_img.shape[2]
            filtered_image = np.zeros_like(image.manipulated_img)
            for ch in range(num_channels):
                filtered_image[:,:,ch] = self.median_2d(image.manipulated_img[:,:,ch], filter_size)
            return filtered_image
        else:
            return self.median_2d(image.manipulated_img, filter_size)
        
    def compute_bilateral_2d(self, image, filter_size, sigma_space, sigma_color):
        height, width = image.shape
        pad_size = filter_size // 2
        
        padded_image = self.padding_image(image, width, height, pad_size)

        spatial_kernel = self.get_gaussian_mask(filter_size, sigma_space)
        color_kernel = lambda difference: np.exp(-(difference**2) / (2 * sigma_color ** 2)) / (np.sqrt(2 * np.pi) * sigma_color)

        filtered_image = np.zeros_like(image)
        for x in range(width):
            for y in range(height):
                neighbourhood = padded_image[y: y + filter_size, x:x + filter_size] 
                color_difference = abs(neighbourhood - padded_image[y + pad_size, x + pad_size])
                brightness_kernel = color_kernel(color_difference) # this kernal changes according to the brightness difference, which makes this filter non-linear  

                weighted_sum = np.sum(neighbourhood * spatial_kernel * brightness_kernel)
                weight_sum = np.sum(spatial_kernel * brightness_kernel)

                filtered_image[y,x] = weighted_sum / weight_sum 

        return filtered_image


    def bilateral(self, image, filter_size, **kwargs):
        sigma_color = kwargs.get("sigma_color", 10)
        sigma_space = kwargs.get("sigma_space", 2)

        if sigma_color == 0  or sigma_space == 0:
            raise ValueError("Sigma value can not be zero")
        if image.colored:
            num_channels = image.manipulated_img.shape[2]
            filtered_image = np.zeros_like(image.manipulated_img)
            for ch in range(num_channels):
                filtered_image[:,:,ch] = self.compute_bilateral_2d(image.manipulated_img[:,:,ch],filter_size, sigma_space, sigma_color )
        else:
            filtered_image = self.compute_bilateral_2d(image.manipulated_img,filter_size, sigma_space, sigma_color)  
            
        return filtered_image

# ---------------------------------------------  Edge Detector ----------------------------------------------------------------------
    

class EdgeDetector:
    def __init__(self) -> None:
        self.lookup = {
            "sobel_3x3": self.sobel_3x3,
            "sobel_5x5": self.sobel_5x5,
            "roberts": self.roberts,
            "prewitt": self.prewitt,
            "laplacian": self.laplacian, 
            "canny": self.canny
        }
        self.filter = ImageFilter() 
        self.edge_detectors = self.lookup.keys()

    def apply_detector(self, image, edge_detector):
        if edge_detector.lower() not in self.edge_detectors:
            raise ValueError(f"Invalid filter type. Allowed types: {', '.join(self.edge_detectors)}")
        image.convert_to_grayscale()
        filtered_image = self.filter.apply_filter(image, "gaussian", 5)  
        image.undo_action()
        return self.lookup[edge_detector.lower()](filtered_image)
    
    def padding_image(self, image, width, height, pad_size):
        padded_image = np.zeros((height + 2 * pad_size, width + 2 * pad_size)) 
        padded_image[pad_size:-pad_size, pad_size:-pad_size] = image 
        return padded_image
    
    def convolve_2d(self, image, kernel, mutlipy = True):
        image_height, image_width = image.shape
        kernel_size = kernel.shape[0]

        pad_size = kernel_size // 2
 
        
        if pad_size == 0:
            padded_image = image
            normalize_value = 2
        else:
            # padding the image to include edges 
            normalize_value = kernel_size * kernel_size 
            padded_image = self.padding_image(image, image_width, image_height, pad_size)
            
        output_image = np.zeros_like(image)
        
        for i in range(image_height):
            for j in range(image_width):
                neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size] # slice out the region 
                # optimization trick  
                if mutlipy:
                    output_image[i, j] = np.sum(neighborhood * kernel)
                else:
                    output_image[i, j] = np.sum(neighborhood) * (1/normalize_value)
        return  np.clip(output_image,0,255)
    
    def compute_gradient_using_convolution(self, image, x_kernel, y_kernel,cart_coord):
        x_component = self.convolve_2d(image,x_kernel)
        y_component = self.convolve_2d(image, y_kernel)
        resultant = np.abs(x_component) + abs(y_component)
        resultant = resultant / np.max(resultant) * 255
        direction = np.arctan2(y_component, x_component) 
        if not cart_coord:
            direction = (direction + np.pi/4) % np.pi  # Wrap angles back to [0, π]
        return (resultant, direction)
    
    def get_edges_using_convolution(self,image, x_kernel, y_kernel, cart_coord = True):
        edged_image = self.compute_gradient_using_convolution(image, x_kernel, y_kernel,cart_coord )
        return edged_image 

    def sobel_3x3(self,image):
        dI_dX = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_using_convolution(image, dI_dX, dI_dY)

    def sobel_5x5(self,image):
        dI_dX = np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-1,-2,0,2,1],[-2,-3,0,3,2]])
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_using_convolution(image, dI_dX, dI_dY)  
    
    def roberts(self, image):
        secondary_diag = np.array([[0,1],[-1,0]])
        main_diag = np.rot90(secondary_diag)
        return self.get_edges_using_convolution(image, secondary_diag, main_diag, False)  

    def prewitt(self, image):
        dI_dX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        dI_dY = np.rot90(dI_dX)
        return self.get_edges_using_convolution(image, dI_dX, dI_dY)  


    def laplacian(self, image):
        kernel = np.array([[1,4,1],[4,-20,4],[1,4,1]])
        edged_image = self.convolve_2d(image, kernel, True)  
        return (edged_image)
    
    def non_maximum_suppression(self, magnitude, direction):
        image_height, image_width = magnitude.shape
        suppressed_image = np.zeros((image_height, image_width), dtype=np.uint8)
        angles = direction * 180 / np.pi
        angles[angles < 0] += 180

        for i in range(1, image_height - 1):
            for j in range(1, image_width - 1):
                q, r = 255, 255  

                if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                    r = magnitude[i, j - 1]
                    q = magnitude[i, j + 1]

                elif (22.5 <= angles[i, j] < 67.5):
                    r = magnitude[i - 1, j + 1]
                    q = magnitude[i + 1, j - 1]

                elif (67.5 <= angles[i, j] < 112.5):
                    r = magnitude[i - 1, j]
                    q = magnitude[i + 1, j]

                elif (112.5 <= angles[i, j] < 157.5):
                    r = magnitude[i + 1, j + 1]
                    q = magnitude[i - 1, j - 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed_image[i, j] = magnitude[i, j]
                else:
                    suppressed_image[i, j] = 0

        return suppressed_image
    
    def threshold(self,img, lowThresholdRatio=0.05, highThresholdRatio=0.2):

        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;
        
        image_height, image_width = img.shape
        res = np.zeros((image_height, image_width), dtype=np.int32)
        
        weak = np.int32(25)
        strong = np.int32(255)
        
        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold) # left zeros 
        
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
        
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        
        return (res, weak, strong)
    
    def hysteresis(self, img, weak, strong=255):
        image_height, image_width = img.shape

        for i in range(1, image_height-1):
            for j in range(1, image_width-1):
                if (img[i, j] == weak):  # these weak edges are considered to be strong, if they are connected to strong edges
                    if (
                        (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                        (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                        (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                        (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                    ):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    def canny(self,image):
        edged_image = self.sobel_3x3(image)
        suppressed_image = self.non_maximum_suppression(edged_image[0],edged_image[1])
        thresholded_image_info = self.threshold(suppressed_image) 
        output_image = self.hysteresis(*thresholded_image_info) 
        return output_image
        
        
# ------------------------------------------------------------------- Image --------------------------------------------
    
class Image:
    def __init__(self, image_path=None, image_data=None):
        if image_path:
            self.original_img = plt.imread(image_path)
        elif image_data is not None:
            self.original_img = image_data
        else:
            raise ValueError("Please provide either image_path or image_data.")

        self.manipulated_img = np.copy(self.original_img)
        self.previous_img = np.array([])
        self.colored = True if len(self.original_img.shape) == 3 else False

    def convert_to_grayscale(self):
        if self.colored:
            self.previous_img = np.copy(self.manipulated_img)
            self.manipulated_img = np.dot(self.manipulated_img[..., :3], [0.2989, 0.5870, 0.1140])
            self.colored = False

    def display(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.manipulated_img, cmap="gray" if not self.colored else None)
        plt.axis('off')
        plt.show()

    def reset_image(self):
        self.manipulated_img = np.copy(self.original_img)
        self.colored = True if len(self.original_img.shape) == 3 else False

    def plot_intensity_histogram(self):
        plt.figure(figsize=(10, 5))
        if not self.colored:
            plt.hist(self.manipulated_img.flatten(), bins=256, color='gray', density=True)
            plt.title('Grayscale Image Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
        else:
            for i, color in enumerate(['red', 'green', 'blue']):
                plt.hist(self.manipulated_img[:, :, i].flatten(), bins=256, color=color, alpha=0.5,
                         label=color.capitalize(), density=True)
            plt.title('Colored Image Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend(bbox_to_anchor=(1.1, 0.5))
        plt.show()

    def adjust_contrast(self, factor):
        self.manipulated_img = (self.manipulated_img - 128) * factor + 128
        self.manipulated_img = np.clip(self.manipulated_img, 0, 255).astype(np.uint8)

    def adjust_brightness(self, delta):
        self.manipulated_img = np.clip(self.manipulated_img + delta, 0, 255).astype(np.uint8)

    def undo_action(self):
        if np.any(self.previous_img):
            self.manipulated_img = np.copy(self.previous_img)
            self.colored = True if len(self.manipulated_img.shape) == 3 else False

# ---------------------------------------------------------- Image Processor -----------------------------------
                

class ImageProcessor:
    def __init__(self) -> None:
        self.__initialize_tools()

    def __initialize_tools(self):
        self.noise_source =  NoiseAdder()
        self.filter = ImageFilter()
        self.edge_detector = EdgeDetector()

    def add_noise(self, image ,noise_type: str = "uniform", strength = 1 ): # strength represent a slider (0 --> 100) 
        noisy_image = self.noise_source.apply_noise(image.manipulated_img, noise_type, strength) 
        image.previous_img = image.manipulated_img 
        image.manipulated_img = noisy_image

    def apply_filter(self, image, filter_type: str = "average", filter_size = 5, **kwargs):
        filtered_image = self.filter.apply_filter(image, filter_type, filter_size, **kwargs)
        image.previous_img = image.manipulated_img 
        image.manipulated_img = filtered_image

    def get_edges(self, image, detector_type: str = "roberts"):
        edged_image  = self.edge_detector.apply_detector(image, detector_type) # edged_image == [mag, direction]
        if len(edged_image) == 2:
            self.plot_gradient_with_orientation(edged_image)
            return edged_image[0]    
        else:
            self.plot_gradient_mag(edged_image)
            return edged_image 

    def plot_gradient_with_orientation(self, image):
        mag = image[0]
        direction = image[1]
        direction_normalized = direction / np.pi
        image_shape = image[0].shape 
        hue = direction_normalized * 360
        hsv_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        hsv_image[..., 0] = hue.astype(np.uint8)
        hsv_image[..., 1] = mag.astype(np.uint8)
        hsv_image[..., 2] = mag.astype(np.uint8)
        rgb_image = mcolors.hsv_to_rgb(hsv_image / 255)  
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_image, cmap="gray")
        plt.axis('off')
        plt.show()


    def plot_gradient_mag(self,image):
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap="gray")
        plt.axis('off')
        plt.show()
        
    def get_noise_types(self):
        print(list(self.noise_source.noise_types))

    def get_filter_types(self):
        print(list(self.filter.filter_types))

    def get_detector_types(self):
        print(list(self.edge_detector.edge_detectors))
    

