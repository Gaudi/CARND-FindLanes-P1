import os
import math
import cv2
import numpy as np
import matplotlib.image as mpimg

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_image
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
def gradient(x1, y1, x2, y2):
    """
       'x1' Cooordinate in the x axis at the start of the line
       'x2' Cooordinate in the x axis at the end of the line
       'y1' Cooordinate in the y axis at the start of the line
       'y2' Cooordinate in the y axis at the end of the line
       Calculate the gradient of the line x1,y1 - x2,y2
    """
    grad = 0
    
    if(x2 == x1):
        grad = (y2-y1)
    else:
        grad = (y2-y1)/(x2-x1)

    return grad


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Create a straight lines with color and thicknes connecting the lines from lines array by selecting 
    them according to their gradient and draw the resulting lines, left and right, in img.
    Lane lines are assumed to have an angle between 25 and 65 degree
    lines considered part of the left lane line are between -25 and -65 degree angle (y is inverted)
    lines considered part of the right lane line are between 25 and 65 degree angle
    Then use all the points for either left/right line to find the line that better describes the whole set.
    AFter the gradient and y-interceptor are calculates using numpy polyfit function,
    use the line equation to obtein the values for x1-x2. The valeus for y are:
    y1 = image height
    y2 = 60% of image height from top to bottom.
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    Once the lines are calculated the line with color and thicknes is draw in img
    """
    image_height = img.shape[0]
    image_width = img.shape[1]
    y_max_height = image_height * (0.60)
    left_X_list = list()
    left_Y_list = list()
    right_X_list = list()
    right_Y_list = list()

    for line in lines:
        for x1,y1,x2,y2 in line:
            grad = gradient(x1,y1,x2,y2)
            degree = math.degrees(math.atan(grad))
            if (degree < -25 and degree > -65) and x1 < (image_width/2):
                left_X_list.append(min(x1,(image_width/2)))
                left_X_list.append(min(x2,(image_width/2)))
                left_Y_list.append(y1)
                left_Y_list.append(y2)
            elif (degree > 25 and degree < 65) and x1 > (image_width/2):
                right_X_list.append(max(x1,(image_width/2)))
                right_X_list.append(max(x2,(image_width/2)))
                right_Y_list.append(y1)
                right_Y_list.append(y2)
    #calculate left line    
    left_coefs = np.polyfit((left_X_list), (left_Y_list), 1)
    bottom_letf_x = int((image_height-left_coefs[1])/left_coefs[0])
    top_letf_x = int((y_max_height-left_coefs[1])/left_coefs[0])
    cv2.line(img, (bottom_letf_x, image_height), ( top_letf_x, int(y_max_height)), color, thickness)
    #calculate right line
    right_coefs = np.polyfit((right_X_list), (right_Y_list), 1)
    bottom_right_x = int((image_height-right_coefs[1])/right_coefs[0])
    top_right_x = int((y_max_height-right_coefs[1])/right_coefs[0])
    cv2.line(img, (bottom_right_x, image_height), ( top_right_x, int(y_max_height)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
    

def process_image(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    gray_image = grayscale(image)

    #lets redcue some detail
    gsblur_gray_image = gaussian_blur(gray_image, 3)

    #Thresholds were selected by trial-error
    low_threshold = 40
    high_threshold =  120
    #lets find the edges oof the image
    edges_image = canny(gsblur_gray_image, low_threshold, high_threshold)

    #define a region to search for lane lines
    #The camera is always at the center of the veicle, so by trial and error again,
    #we define how far from the center the lines must be found
    #for the width dividing the image in chunks of 18 pixels seems to fit ok
    #for height 315 seemed an ok value
    imgWidth = image.shape[1]
    imgHeight = image.shape[0]
    edge_heigh = imgHeight * (0.60)
    left_bottom = [int(imgWidth/8), imgHeight]
    right_bottom = [imgWidth, imgHeight]
    left_top_poly = [ (imgWidth/2) - 50, edge_heigh]
    right_top_poly = [imgWidth/2 + 50, edge_heigh]
    
    right_bottom_2 = [int(imgWidth*7/8) , imgHeight]
    right_top_poly_2 = [ imgWidth/2 , edge_heigh+20]
    left_top_poly_2 = [ (imgWidth/2) - 20, edge_heigh+20]
    left_bottom_2 = [int(imgWidth*2/7) , imgHeight]
    vertices = np.array([[left_bottom, left_top_poly, right_top_poly, right_bottom, right_bottom_2, right_top_poly_2, left_top_poly_2, left_bottom_2, left_bottom ]],dtype=np.int32)

    mask_edges = region_of_interest(edges_image, vertices)

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 12    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 #minimum number of pixels making up a line
    max_line_gap = 10   # maximum gap in pixels between connectable line segments

    hough_img = hough_lines(mask_edges, rho, theta, threshold, min_line_length, max_line_gap)

    org_image = np.copy(image)

    hough_lines_image = gaussian_blur(hough_img, 13)

    weighted_image = weighted_img(org_image, hough_lines_image,0.8 , 1.0, 0.0)
    
    return weighted_image


def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        # directory already exists
        pass
    

def process_directory(input_path, output_path):
    """
        'inputh_path' is the directory path of the images to be processed
        'output_path' is the directory path where images will be saved after processed by the pipleine
        The function iterates for all the images in input_path, processed them throug the pipelien to find
        lane lines, and store the resulting images in output_path
    """
    filename = []
    
    create_dir(input_path)
    create_dir(output_path)

    for rootdir, directory, files in os.walk(input_path):
        for file in files:
            if '.jpg' in file:
                image = mpimg.imread(input_path+file)
                region_image = np.copy(image)
                processed_image = process_image(region_image)  
                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)        
                cv2.imwrite(output_path+file, rgb_image)
            elif '.mp4' in file:
                process_video(input_path+file, output_path+file)
        
    print('Processed all files in folder \'', input_path, '\' has been processed and placed in \'', output_path, '\'' )

    
def process_video(input_video_path, output_video_path):
    """
        'input_video_path' is the directory path of the videos to be processed
        'output_video_path' is the directory path where videos will be saved after processed by the pipleine
        The function iterates for all the videos in input_video_path, processed them throug the pipelien to find
        lane lines, and store the resulting videos in output_video_path
    """
    cap = cv2.VideoCapture(input_video_path)
    ret = True
  
    #lets use the same properties as the original video
    if cap.isOpened():
        width = int(cap.get( cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    else: return
    
    #select a codec for the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height),1)

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == False:
            break
        #pipeline process images in RGB, lets please it
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_img = process_image(bgr_image)
        #Our Video needs BGR format (is this true???)
        video_out.write(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    cap.release()
    cv2.destroyAllWindows()
        
        
   

