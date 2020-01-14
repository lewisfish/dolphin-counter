import sys

import numpy as np
from matplotlib import pyplot as plt

from skimage import io
from skimage.color import rgb2ycbcr
from skimage.filters import threshold_yen
from skimage.measure import regionprops, label
from skimage.morphology import square, opening

from skimage.util import img_as_ubyte
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage.draw import ellipse

file = "2.png"
img = io.imread(file)
img = img[130:1030, 0:1990]
hsv = rgb2ycbcr(img)

data = hsv[:, :, 0]

ny, nx = data.shape
print(data.shape)
y, x = np.mgrid[:ny, :nx]

sigma_clip = SigmaClip(sigma=2.5)
bkg_estimator = MedianBackground()
bkg = Background2D(data, box_size=(10, 10), filter_size=(5, 5),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

bkgsub = data - bkg.background
bkgsub = bkgsub / np.amax(bkgsub)
bkgsub = img_as_ubyte(bkgsub)


thresh = threshold_yen(bkgsub, nbins=256)
binary = data >= 150
thresh = opening(binary, square(1))

labels = label(thresh)
areas = []
mask = np.zeros_like(img)
for region in regionprops(labels):
    a = region.major_axis_length
    b = region.minor_axis_length
    areas.append(region.area)
    if a > 0 and b > 0 and region.area < 50:
        theta = region.orientation
        centre = region.centroid
        r, c = ellipse(centre[0], centre[1], b, a, rotation=1.57+theta)
        r = np.where(r >= ny, ny-1, r)
        c = np.where(c >= nx, nx-1, c)
        mask[r, c] = [255, 0, 0]

plt.imshow(img)
plt.imshow(mask, alpha=0.3)
plt.show()
