# Histogram-Equalization-Image-Quantization
Performing histogram equalization &amp; optimal quantization on Images. 

In this project, as part of Image processing course, I've implemented histogram equalization and quantization on images, based on the follows algorithms:

Histogram equalization Algorithm: 
1. Compute the image histogram.
2. Compute the cumulative histogram.
3. Normalize the cumulative histogram.
4. Multiply the normalized histogram by the maximal gray level value (Z − 1).
5. Verify that the minimal value is 0 and the the maximal is Z−1, otherwise, stretch the result linearly in the range [0,Z − 1].
6. Round the values to get integers.
7. Map the intensity values of the image using the result of step 6.
Let C(k) be the cumulative histogram at intensity k and let m be the first gray level for which C(m) ̸= 0, then the transformation T is: T(k) = round[255 · C(k) − C(m) / (C(255) − C(m))]

For example, for a simple black image (left image), we will get the right image: ![left right](https://user-images.githubusercontent.com/64755588/205483921-5665af02-d92f-46f5-aa15-9f392d81c44e.png)


Optimal image quantization Algorithm: 
![q_algo](https://user-images.githubusercontent.com/64755588/205484029-53404a27-bf24-4ab0-b89a-4069c77c93e1.png)
Note about the algorithm: 
If an RGB image is given, the quantization procedure should only operate on the Y channel of the
corresponding YIQ image and then convert back from YIQ to RGB. Each iteration of the quantization
process is composed of the following two steps:
• Computing z - the borders which divide the histograms into segments. z is an array with shape
(n_quant+1,). The first and last elements are 0 and 255 respectively.
• Computing q - the values to which each of the segments’ intensities will map. q is also a one
dimensional array, containing n_quant elements.

For example, for the simple black image from last algorithm: 
![q_res](https://user-images.githubusercontent.com/64755588/205484095-2bee13c9-f61a-466d-8744-891b54ab4b01.png)
