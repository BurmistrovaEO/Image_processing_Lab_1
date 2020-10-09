from PIL import Image, ImageDraw
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import asksaveasfile
import os
import cv2
import math
import numpy as np
import matplotlib


def psnr(img1, img2):
    img1 = cv2.resize(img1, (1000, 1000), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (1000, 1000), interpolation=cv2.INTER_AREA)
    img1 = img1.astype(np.float64) / 255.
    img2 = img2.astype(np.float64) / 255.
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return "100"
    return 10 * math.log10(1. / mse)


def bright(img1, brightness):
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            (b, g, r) = img1[x, y]

            red = int(r * brightness)
            red = min(255, max(0, red))

            green = int(g * brightness)
            green = min(255, max(0, green))

            blue = int(b * brightness)
            blue = min(255, max(0, blue))

            img1[x, y] = (blue, green, red)
    return img1


def file_open_general():
    f = askopenfilename(title="Select file", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"),
                                                        ("jpeg files", "*.jpeg")))
    if f:
        return Image.open(f)


def file_open_cv():
    f = askopenfilename(title="Select file", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"),
                                                        ("jpeg files", "*.jpeg")))
    if f:
        return cv2.imread(f, cv2.IMREAD_COLOR)


def file_save(im):
    f = asksaveasfile(mode='w', defaultextension=".png")
    if f:  # asksaveasfile return `None` if dialog closed with "cancel".
        abs_path = os.path.abspath(f.name)
        im.save(abs_path)
    f.close()


def gray_scale_filter_pil():
    image = file_open_general()
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    for i in range(width):
        for j in range(height):
            r = pix[i, j][0]
            g = pix[i, j][1]
            b = pix[i, j][2]
            s = round(0.2126 * r + 0.7152 * g + 0.0722 * b)
            draw.point((i, j), (s, s, s))
    image.show()
    file_save(image)


def gray_scale_filter_cv():
    cv2.startWindowThread()
    img = file_open_cv()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Press esc to close')
    cv2.imshow('Press esc to close', gray_image)
    gray_image = Image.fromarray(gray_image)
    file_save(gray_image)
    # cv2.waitKey()
    cv2.destroyAllWindows()


def convert_to_hsv_and_back():
    image = file_open_general()
    im_array = np.array(image)
    print(type(im_array[0]))
    im_array = matplotlib.colors.rgb_to_hsv(im_array)
    im = Image.fromarray((im_array * 255).astype(np.uint8))
    im.show()

    img = image.convert(mode="RGB")
    img.show()


def convert_to_hsv_and_back_cv():
    cv2.startWindowThread()
    image = file_open_cv()
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.namedWindow('Press esc to close(hsv)')
    cv2.imshow('Press esc to close(hsv)', hsv)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    cv2.namedWindow('Press esc to close(rgb)')
    cv2.imshow('Press esc to close(rgb)', bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()


print("Choose preferred mode:\n"
      "1 - Peak signal-to-noise ratio\n"
      "2 - Gray Scale filter(PIL)\n"
      "3 - Gray Scale filter(OpenCV)\n"
      "4 - Convert color model from RGB to HSV and back(PIL + matplotlib)\n"
      "5 - Convert color model from RGB to HSV and back(OpenCV)\n"
      "6 - Brightness improvement\n")
print('Input mode № = ')
mode = input()
if mode == '1':
    image_1 = file_open_cv()
    image_2 = file_open_cv()
    d = psnr(image_1, image_2)
    print("Изображения схожи на", d, "%")
elif mode == '2':
    gray_scale_filter_pil()
elif mode == '3':
    gray_scale_filter_cv()
elif mode == '4':
    convert_to_hsv_and_back()
elif mode == '5':
    convert_to_hsv_and_back_cv()
elif mode == '6':
    cv2.startWindowThread()
    image_1 = file_open_cv()
    b = bright(image_1, 2)
    cv2.namedWindow('Press esc to close')
    cv2.imshow("Press esc to close", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
