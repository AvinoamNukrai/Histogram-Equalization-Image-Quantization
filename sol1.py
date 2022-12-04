# Written by Avinoam Nukrai - Ex1 of IMPR course, Hebrew U 2023

import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
from skimage.color import rgb2gray
import scipy
import skimage.color

GRAYSCALE = 1
RGB = 2
RGB_DIM = 3
MAX_GS = 255
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # We have 3 options only of converting: rgb -> rgb, gs -> gs, rgb -> gs
    image = iio.imread(filename)
    if (len(image.shape) == RGB_DIM and representation == RGB) or \
            (len(image.shape) != RGB_DIM and representation == GRAYSCALE):
        # rgb -> rgb or gs -> gs
        return np.float64(image / MAX_GS)
    elif len(image.shape) == RGB_DIM and representation == GRAYSCALE:
        # rgb -> gs
        return rgb2gray(image)


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)
    plt.imshow(image, interpolation="nearest", cmap='viridis' if (representation == RGB) else 'gray')
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    yiq_image = imRGB @ np.transpose(RGB_YIQ_TRANSFORMATION_MATRIX)
    return yiq_image.reshape(imRGB.shape)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    inv = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
    return (imYIQ @ np.transpose(inv)).reshape(imYIQ.shape)


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # check the type of the input img
    if len(im_orig.shape) != RGB_DIM:  # grayscale
        return histogram_equalize_algo(im_orig)
    else:  # RGB
        yiq_img = rgb2yiq(im_orig)
        Y = yiq_img[:, :, 0]
        eq_hist = histogram_equalize_algo(Y)
        yiq_img[:, :, 0] = eq_hist[0]
        return [yiq2rgb(yiq_img), eq_hist[1], eq_hist[2]]


def calc_lookup_table(cum_histogram):
    """
    This function calculate the lookup table (T) that was defined in the ex
    :param cum_histogram: the cumulated histogram
    :return: lookup table
    """
    max_gs_level = np.argmax(cum_histogram > 0)  # Find the index of max gs level elem
    return np.array([int(MAX_GS * ((cum_histogram[k] - cum_histogram[max_gs_level]) /
                            (cum_histogram[MAX_GS] - cum_histogram[max_gs_level])))
              for k in range(len(cum_histogram))])


def histogram_equalize_algo(img):
    """
    This function ia the algorithm of histogram_equalize of some given img
    :param img: image to conduct the histogram_equalization on
    :return: equalize histogram of img
    """
    img = img * MAX_GS  # We need to mult img in 255 for being able to see the histogram
    orig_histogram, bins = np.histogram(img, bins=MAX_GS + 1, range=[0, MAX_GS], normed=None, weights=None, density=None)  # Compute the image histogram
    cum_histogram = np.cumsum(orig_histogram)  # Compute the cumulative histogram
    image_equalize = calc_lookup_table(cum_histogram)[img.astype(np.int64)].astype(np.float64)
    # image_equalize is matrix in which for every elem in image image_equalize is T[elem] with converting back to float
    equalize_histogram, bins = np.histogram(image_equalize, bins=MAX_GS+1, range=[0, MAX_GS], normed=None, weights=None, density=None)
    image_equalize /= MAX_GS  # normalize the result image
    return [image_equalize, orig_histogram, equalize_histogram]


def error_calc_per_iter(z_array, q_array, n_quant):
    """
    This function calculates the error of some partition of z and q axis
    :param z_array: the given z array to calc on
    :param q_array: the given q array to calc on
    :param n_quant: number of qountization we want to do
    :return: the error array of the specific given partition
    """
    errors = np.arange(MAX_GS + 1).astype(float)
    for i in range(n_quant):
        lower_bound = np.floor(z_array[i]).astype(int)
        upper_bound = np.floor(z_array[i + 1]).astype(int)
        if len(errors) <= 1:
            break
        errors[lower_bound:upper_bound] -= q_array[i]
    errors[MAX_GS] -= q_array[n_quant - 1]
    return errors


def find_z_from_q(z_array, q_array, n_quant):
    """
    This function calculates z array from a given q array
    :param z_array: prev z array in the partition (or the first one)
    :param q_array: q array to calc on
    :param n_quant: number of quantization we want to do
    :return: the updated z array
    """
    for i in range(1, n_quant):
        z_array[i] = (q_array[i - 1] + q_array[i]) / 2
    return z_array


def find_q_from_z(z_array, q_array, n_quant, norm_hist):
    """
    This function calculates q array from a given z array
    :param z_array: array to calc on
    :param q_array: prev q array to update
    :param n_quant: number of quant we want to do
    :param norm_hist: the normalize histogram
    :return: the updated q array
    """
    for i in range(n_quant):
        lower_bound = np.floor(z_array[i]).astype(int)
        upper_bound = np.floor(z_array[i + 1]).astype(int)
        weights = norm_hist[lower_bound:upper_bound]
        if np.sum(weights) == 0:
            q_array[i] = 0
        else:
            q_array[i] = np.sum(np.arange(lower_bound, upper_bound) * weights) / np.sum(weights)
    return q_array


def calc_quantize_img(orig_img, img_type, yiq_img, orig_hist, n_quant, z_array, q_array):
    """
    This function calculates the final quantized image
    :param orig_img:
    :param img_type: the type of the orig image (RGB or Grayscale)
    :param yiq_img:
    :param orig_hist:
    :param n_quant:
    :param z_array:
    :param q_array:
    :return: the final image we want to plot
    """
    optimal_img = orig_hist
    # map the new img into optimal_img
    for i in range(n_quant):
        lower_bound = np.floor(z_array[i]).astype(int)
        upper_bound = np.floor(z_array[i + 1]).astype(int)
        optimal_img[lower_bound:upper_bound] = q_array[i]
    optimal_img[MAX_GS] = q_array[n_quant - 1]
    optimal_img = optimal_img[orig_img.astype(int)]
    # checking if need to output according RGB image or GRAYSCALE
    if img_type != RGB:
        optimal_img = optimal_img / MAX_GS
    else:
        new_im_yiq = np.dstack((optimal_img / MAX_GS, yiq_img[:, :, 1], yiq_img[:, :, 2]))
        optimal_img = yiq2rgb(new_im_yiq)
    return optimal_img


def initial_z_q_partition(cum_hist, n_quant):
    """
    This function calcs the first partition of z and q axis
    :param cum_hist: the cum histogram
    :param n_quant: number of quant
    :return: the initial partition
    """
    z_arr, q_arr = [0], [0] * n_quant
    factor = cum_hist[MAX_GS] // n_quant
    for i in range(1, n_quant):
        z_arr.append(np.where(cum_hist > i * factor)[0][0])
    z_arr.append(MAX_GS)
    return q_arr, z_arr


def quantize_image_algo(im_orig, yiq_img, n_quant, n_iter):
    """
    This function implements the quantize algorithm we've learned
    :param im_orig: the original image
    :param yiq_img: the YIQ image
    :param n_quant: nuber of quant
    :param n_iter: number of iterations we want to iterate over for finding the optimal partition
    :return: array of [final quantized image, array of errors]
    """
    orig_hist, bins = np.histogram(im_orig, bins=np.arange(MAX_GS + 2))
    norm_hist = orig_hist / np.sum(orig_hist)
    q_array, z_array = initial_z_q_partition(np.cumsum(orig_hist), n_quant)
    errors, errors_len = np.zeros(n_iter), n_iter
    for i in range(n_iter):
        q_array = find_q_from_z(z_array, q_array, n_quant, norm_hist)  # find q_arr from z
        z_array = find_z_from_q(z_array, q_array, n_quant)  # find z_arr from q
        error_per_iter = error_calc_per_iter(z_array, q_array, n_quant)  # calc the error
        errors[i] = np.sum(np.multiply(np.square(error_per_iter), norm_hist))
        if errors[i] == errors[i - 1] and i > 0:  # check if need to stop iterate
            errors_len = i
            break
    errors = errors[0:errors_len]
    img_type = RGB if (len(im_orig.shape) == RGB_DIM) else GRAYSCALE
    final_im = calc_quantize_img(im_orig, img_type, yiq_img, np.copy(orig_hist), n_quant, z_array, q_array)  # create the final image
    return [final_im, errors]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    # first, need to check the type of the image
    if len(im_orig.shape) != RGB_DIM:  # grayscale image
        img_origi_mult = (im_orig * MAX_GS).astype(int)
        return quantize_image_algo(img_origi_mult, None, n_quant, n_iter)
    else:  # rgb image
        yiq_img = rgb2yiq(im_orig)
        Y = (yiq_img[:, :, 0] * MAX_GS).astype(int)
        data = quantize_image_algo(Y, yiq_img, n_quant, n_iter)
        yiq_img[:, :, 0] = data[0]  # updating new Y values
        return [yiq2rgb(yiq_img), data[1]]


def toy_image():
    """
    This func calc the toy img that has given in the ex
    :return: the matrix (image)
    """
    x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
                   np.array([255] * 6)[None, :]])
    grad = np.tile(x, (256, 1))
    return grad


if __name__ == "__main__":
    toy = toy_image()
    # print(np.empty([10]).shape)
    # imdisplay("jerusalem.jpg", 2)
    # im = read_image("jerusalem.jpg", 2)
    # print(im.shape)
    # avi = rgb2yiq(im)
    # avi_2 = yiq2rgb(avi)
    # plt.imshow(avi_2)
    # plt.show()