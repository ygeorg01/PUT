import cv2
import sys, os
import numpy as np
# def distortions(im):

# def black_blobs(im):

def gaussian_noise(img, noise_typ="gauss"):

	# print("in gauss noise: ", noise_typ)
	if noise_typ == "gauss":
		# print("in noise typ")
		row,col,ch= img.shape
		mean = 0
		var = 5
		sigma = var
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		# print("gauss: ", gauss)
		gauss = gauss.reshape(row,col,ch)
		noisy = img + gauss
	
		return noisy
  #  elif noise_typ == "s&p":
  #     row,col,ch = image.shape
  #     s_vs_p = 0.5
  #     amount = 0.004
  #     out = np.copy(image)
  #     # Salt mode
  #     num_salt = np.ceil(amount * image.size * s_vs_p)
  #     coords = [np.random.randint(0, i - 1, int(num_salt))
  #             for i in image.shape]
  #     out[coords] = 1

  #     # Pepper mode
  #     num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
  #     coords = [np.random.randint(0, i - 1, int(num_pepper))
  #             for i in image.shape]
  #     out[coords] = 0
  #     return out
  # elif noise_typ == "poisson":
  #     vals = len(np.unique(image))
  #     vals = 2 ** np.ceil(np.log2(vals))
  #     noisy = np.random.poisson(image * vals) / float(vals)
  #     return noisy
  # elif noise_typ =="speckle":
  #     row,col,ch = image.shape
  #     gauss = np.random.randn(row,col,ch)
  #     gauss = gauss.reshape(row,col,ch)        
  #     noisy = image + image * gauss
  #     return noisy

def gaussian_blur(img, kernel_size):

	return cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)

def main():
	directory = "images2_2"

	count = 0
	ordering = True
	kernel = np.ones((2,2),np.uint8)
	for filename in sorted(os.listdir(directory)):
		
		img = cv2.imread(os.path.join(directory,filename), cv2.IMREAD_UNCHANGED)

		img_dist = gaussian_blur(img, 3)

		# laplacian = cv2.Laplacian(img,cv2.CV_64F)
		# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
		# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

		img_dist = gaussian_noise(img_dist).astype(np.uint8)

		# img_dist = cv2.morphologyEx(img_dist, cv2.MORPH_GRADIENT, kernel)
		# print("img_dist: ", img_dist.shape)
		# print(img_dist.dtype)
		# print("img: ", img.shape)
		# print(img.dtype)

		im_h = cv2.hconcat([img, img_dist])
		cv2.imwrite('test.png', im_h)
		break



if __name__ == "__main__":
	main()

