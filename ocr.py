import numpy as np
import os
import cv2
import Image
import ImageFilter
import pytesseract
from math import *
from itertools import chain
from fuzzywuzzy import fuzz

# build character whitelist
lower = "".join([chr(i) for i in range(ord('a'), ord('z') + 1)])
upper = "".join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
numeric = "".join([chr(i) for i in range(ord('0'), ord('9') + 1)])
spec = "(),.-<>"
whitelist = lower + upper + numeric + spec
tess_conf = " ".join(['-c', 'tessedit_char_whitelist=' + whitelist])

# custom config file locations
base_dir = "C:\\Users\\vge2\\Documents\\"
mixed_char_config = base_dir + "mixed_char"
non_dict_config = base_dir + "non_dict"
small_words_config = base_dir + "small_words"

# list of headers, semantically grouped
headers = ["IEMA Ref#", "UIUC Inv #", "Building", "Room", "Laser manufacturer",
           ["Model", "Laser Model"], "Serial number", "Laser class",
           ["Type", "Laser Type"],
           "Lasing medium", "Operable?",
           "Wavelengths (nm)", "Max Power (W)", "Pulse duration (nsec)",
           "Pulse frequency (MHz)", "Emerging beam divergence (mrad)",
           "Beam diameter (cm)"]
all_headers = list(chain(*[hs if type(hs) is list else [hs] for hs in headers]))
# TODO: add all headers

# headers with special properties
# TODO: more characteristics
mixed_char_headers = ["IEMA Ref#", "UIUC Inv #", "Room"]
single_char_headers = ["Laser class"]
single_block_headers = ["Emerging beam divergence (mrad)"]
non_dict_headers = ["IEMA Ref#", "UIUC Inv #", "Room", "Lasing medium"]

# color constants (BGR format)
white = (255, 255, 255)
blue = (255, 0, 0)
red = (0, 0, 255)

# template image
report_template = cv2.imread("laser_safety_report.png")

# utility functions

def show(image):
    cv2.imshow("image", image); cv2.waitKey()

def write_lines(im, lines, color = (0, 0, 255), thick = 2):
    """write a set of lines in rho, theta form to image"""
    for rho,theta in lines:
        ((x1, y1), (x2, y2)) = rho_to_xy(rho, theta)
        cv2.line(im, (x1,y1), (x2,y2), color, thick)

def print_table(table):
    col_widths = [max([max(map(len, table[j][i].splitlines()))
                       for j in range(len(table))])
                  for i in range(len(table[0]))]
    row_heights = [max([len(entry.splitlines()) for entry in row])
                   for row in table]
    separator = "-" * (sum(col_widths) + (len(col_widths) + 1) * 3 - 2)
    separator = " " + separator + " "

    print(separator)
    for row, height in zip(table, row_heights):
        for row_level in range(height):
            out = " | "
            for entry, width in zip(row, col_widths):
                if row_level < len(entry.splitlines()):
                    s = entry.splitlines()[row_level]
                else:
                    s = ""
                lpad = (width - len(s)) // 2
                rpad = (width - len(s)) - lpad
                out += " " * lpad + s + " " * rpad + " | "
            print(out)
        print(separator)

def lazy_map(func, seq):
    for x in seq:
        yield func(x)

# line conversion functions

def rho_to_xy(rho, theta):
    """convert line from rho, theta to two points that define the line"""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    x1 = x0 + 2000*(-b)
    y1 = y0 + 2000*(a)
    x2 = x0 - 2000*(-b)
    y2 = y0 - 2000*(a)
        
    return ((int(x1), int(y1)), (int(x2), int(y2)))

def xy_to_mb(pt1, pt2):
    """convert two points of line into y = mx + b form.
       if line is vertical, return (None, x-intercept)"""
    (x1, y1) = pt1
    (x2, y2) = pt2
    if abs(x1 - x2) > 0.001:
        m = float(y1 - y2) / (float(x1 - x2) + 0.0000001)
        b = y1 - m * x1
    else:
        m, b = None, x1
    return m, b

def xy_to_rho(pt1, pt2):
    """convert two points of line into rho, theta form"""
    # find inverse of slope of line
    m, b = xy_to_mb(pt1, pt2)
    minv = -1 / m if m else None if m == 0 else 0

    # find intersection point of line with line defined by rho, theta
    intersection = line_intersection(m, b, minv, 0)

    # rho is distance of line to origin
    rho = dist(intersection, (0, 0))

    # theta is angle of perpendicular line with positive x-axis
    theta = atan2(intersection[1], intersection[0])
    return rho, theta

# line manipulation functions

def clamp_xy_to_border(im, pt1, pt2):
    """slide two points of line to be within borders of image"""
    # define the borders of image
    height, width, _ = im.shape
    lft_bot, lft_top = ((0, 0), (0, height))
    rht_top, rht_bot = ((width, height), (width, 0))

    # find intersections of line with borders of image
    lft_intersect = line_intersection_xy((pt1, pt2), (lft_bot, lft_top))
    rht_intersect = line_intersection_xy((pt1, pt2), (rht_bot, rht_top))
    bot_intersect = line_intersection_xy((pt1, pt2), (lft_bot, rht_bot))
    top_intersect = line_intersection_xy((pt1, pt2), (lft_top, rht_top))

    # collect all intersection points that are within borders of image
    bordering_pts = []
    for pt in [lft_intersect, rht_intersect, bot_intersect, top_intersect]:
        # skip point if line is parallel to the axis
        if pt is not all and pt is not None:
            x, y = pt
            if -1 <= x <= width and -1 <= y <= height:
                bordering_pts.append((min(max(x, 0), width - 1),
                                      min(max(y, 0), height - 1)))

    # use intersection points as new points of the line
    if len(bordering_pts) >= 2:
        return tuple(bordering_pts[:2])
    else:
        #print("unable to clamp points of line: " + str(pt1) + ", " + str(pt2))
        return (pt1, pt2)

def iter_neighbors(x, y, max_d = 1):
    """iterate over 3x3 block of points surrounding a point, including point"""
    for dx in range(-max_d, max_d + 1):
        for dy in range(-max_d, max_d + 1):
            yield x + dx, y + dy

def erase_nearest(im, lines, color = (255, 255, 255), thick = 1):
    # TODO: fix
    (height, width, _) = im.shape
    mask = np.zeros(im.shape, dtype = np.uint8)
    for line in lines:
        for x, y in iter_line(line, im):
            # move outward in spiral pattern to find nearest black pixel
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            d = 0
            b = 0
            i = 1
            for _ in range(100):
                if not(0 <= x < width and 0 <= y < height):
                    break
                if im[y][x].sum() < 200:
                    for x2, y2 in iter_neighbors(x, y):
                        mask[y2][x2] = np.array(color, dtype = np.uint8)
                    break
                dx, dy = directions[d]
                x += dx * i; y += dy * i
                d = (d + 1) % 4
                if b == 1:
                    i = i + 1
                    b = 0
                else:
                    b = 1

    # fill in mask
    for x in range(width):
        for y in range(height):
            if mask[y][x].sum() > 0:
                im[y][x] = mask[y][x]
    return mask

def normalize_lines(lines):
    """reflect line so that rho is positive"""
    return np.array([l if l[0] > 0 else (-l[0], -l[1]) for l in lines])

def iter_line_xy(pt1, pt2, im):
    """iterate over all points traversed by the line within image boundaries"""
    (height, width, _) = im.shape
    m, b = xy_to_mb(pt1, pt2)
    if m is not None:
        last_x, last_y = None, None
        num_pts = 1000
        for i in range(num_pts + 1):
            x = int((pt1[0] * i + pt2[0] * (num_pts - i)) / num_pts)
            y = int((pt1[1] * i + pt2[1] * (num_pts - i)) / num_pts)
            if not(x == last_x and y == last_y):
                last_x = x
                last_y = y
                yield x, y
    else:
        for y in range(height):
            yield b, y

def iter_line(line, im):
    """iterate over all points traversed by the line within image boundaries"""
    pt1, pt2 = clamp_xy_to_border(im, *rho_to_xy(*line))
    return iter_line_xy(pt1, pt2, im)

def covers_pixel(x, y, im, tol):
    (height, width, _) = im.shape
    if 0 <= x < width and 0 <= y < height:
        if im[y][x].sum() < tol:
            return True
    return False

def pixels_covered(line, im, neighbors = True, tol = 100):
    """count the number of pixels that a line covers in an image"""
    covers_pixel_ = lambda x, y: covers_pixel(x, y, im, tol)
    criteria = lambda x, y: any([covers_pixel_(x2, y2)
                                 for x2, y2 in iter_neighbors(x, y)]) \
               if neighbors else \
               lambda x, y: covers_pixel_(x, y)
    return sum([1 for x, y in iter_line(line, im) if criteria(x, y)])

def consolidate_lines(lines, im, group_by_intersection = False,
                      sort_by_coverage = True, tol = 300):
    """remove lines that are too similar to one another"""
    (height, width, _) = im.shape
    
    def lines_similar(line1, line2):
        return abs(line2[0] - line1[0]) < 10 and \
               (abs((line2[1] - line1[1]) % np.pi) < 0.1 or \
                abs((line2[1] - line1[1]) % np.pi) > 3.0)

    def intersection_in_bounds(line1, line2):
        pt = line_intersection_polar(line1, line2)
        if pt is None:
            # parallel lines that are not equal do not intersect
            return False
        elif pt is all:
            # parallel lines that are equal always intersect
            return True
        else:
            # otherwise check if intersection point is relatively within bounds
            return -width <= pt[0] < width * 2 and -height <= pt[1] < height * 2

    # aggregate all lines into groups of similar lines
    is_same_group = intersection_in_bounds if group_by_intersection else \
                    lines_similar
    line_groups = []
    for line in lines:
        for group in line_groups:
            if all(lazy_map(lambda member: is_same_group(member, line), group)):
                group.append(line)
                break
        else:
            line_groups.append([line])

    # determine how to select most representative line
    if sort_by_coverage:
        # sort lines by number of black pixels the line covers
        key = lambda l: pixels_covered(l, im, neighbors = True, tol = tol)
    else:
        # sort lines by magnitude of slope (or 10^10 if vertical line)
        key = lambda l: abs(xy_to_mb(*rho_to_xy(*l))[0] or 10 ** 10)

    # find the best line to represent each group of lines
    line_reps = []
    for group in line_groups:
        # pick line that satisfies criteria the best
        if len(group) > 1:
            line_rep = max(group, key = key)
        else:
            line_rep = group[0]
        line_reps.append(line_rep)
    return np.array(line_reps)

def remove_distant_lines(lines, dist_tol = 300):
    """filter out lines which are too far away from others"""
    far_away = lambda line, line2: line_dist(line, line2) > dist_tol
    return filter(lambda line: any([line is not line2 and
                                    not far_away(line, line2)
                                 for line2 in lines]), lines)

def line_intersection(m1, b1, m2, b2):
    """find (x, y) of point intersection of two lines"""
    if m1 is None and m2 is None:
        return all if abs(b1 - b2) < 0.001 else None
    if m1 is None:
        return b1, m2 * b1 + b2
    elif m2 is None:
        return b2, m1 * b2 + b1
    elif abs(m1 - m2) < 0.001:
        return all if abs(b1 - b2) < 0.001 else None
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    return (x, y)

def line_intersection_xy(line1, line2):
    """convert lines to slope, intercept form before finding intersection"""
    m1, b1 = xy_to_mb(*line1)
    m2, b2 = xy_to_mb(*line2)
    return line_intersection(m1, b1, m2, b2)

def line_intersection_polar(line1, line2):
    """convert lines to slope, intercept form before finding intersection"""
    return line_intersection_xy(rho_to_xy(*line1), rho_to_xy(*line2))

def line_dist(line, line2):
    """return distance between lines using rho, or 0 if they intersect"""
    pt = line_intersection_polar(line, line2)
    if pt is None:
        return abs(line[0] - line2[0])
    elif pt is all:
        return 0
    elif (-1000 <= pt[0] < 1000 and -1000 <= pt[1] < 1000):
        return 0
    else:
        return abs(line[0] - line2[0])

def dist(pt1, pt2):
    """return euclidean distance between points"""
    return sqrt(sum([(x1 - x2) * (x1 - x2) for x1, x2 in zip(pt1, pt2)]))

# image manipulation functions

def crop_to_poly(im, pts):
    """crop image to a given polygon, specified as list of points"""
    corners = np.array([pts], dtype = np.int32)
    mask = np.zeros(im.shape, dtype = np.uint8) + 255
    channels = im.shape[2]
    ignore_mask_color = (0,) * channels
    cv2.fillPoly(mask, corners, ignore_mask_color)
    return cv2.bitwise_or(im, mask)

def crop_to_lines(im, horiz_1, horiz_2, vert_1, vert_2):
    """crop to pixels in image that are between horizontal and vertical lines"""
    pt1 = line_intersection_polar(horiz_1, vert_1)
    pt2 = line_intersection_polar(horiz_1, vert_2)
    pt3 = line_intersection_polar(horiz_2, vert_2)
    pt4 = line_intersection_polar(horiz_2, vert_1)
    pts = [pt1, pt2, pt3, pt4]

    # TODO: crop to interior of rectangle, instead of exterior?
    im_crop = crop_to_poly(im, pts)
    min_x = int(max(min([pt[0] for pt in pts]), 0))
    max_x = int(min(max([pt[0] for pt in pts]), im.shape[1]))
    min_y = int(max(min([pt[1] for pt in pts]), 0))
    max_y = int(min(max([pt[1] for pt in pts]), im.shape[0]))
    return im_crop[min_y:max_y, min_x:max_x]

def crop_to_pixels(img):
    """remove empty space between border and non-blank pixels of image"""
    height, width, _ = img.shape
    min_x, max_x, min_y, max_y = 0, width, 0, height

    # find left-most non-blank column of pixels
    for x in range(width):
        if img[:,x].sum() < 3 * height * 250:
            min_x = x
            break

    # find right-most non-blank column of pixels
    for x in range(width - 1, 0, -1):
        if img[:,x].sum() < 3 * height * 250:
            max_x = x
            break

    # find bottom-most non-blank row of pixels
    for y in range(height):
        if img[y,:].sum() < 3 * width * 250:
            min_y = y
            break

    # find top-most non-blank row of pixels
    for y in range(height - 1, 0, -1):
        if img[y,:].sum() < 3 * width * 250:
            max_y = y
            break
    return img[max(0, min_y - 5) : min(height, max_y + 5),
               max(0, min_x - 5) : min(width, max_x + 5)]

def remove_gray(img, low = None, mix = 0.5):
    """denoise image before converting to black and white"""
    img_copy = img.copy()
    cv2.fastNlMeansDenoising(img, img_copy, 30, 7, 21)
    if low is None:
        # use kmeans clustering to find avg value of white, gray, black pixels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = 3, random_state = 1)
        img_px = img_copy.reshape([img_copy.shape[0] * img_copy.shape[1], 3])
        kmeans.fit(img_px)
        centers = list(sorted(map(lambda c: c[0], kmeans.cluster_centers_)))
        low = centers[0] * (1 - mix) + centers[1] * mix
    return cv2.threshold(img_copy, low, 255, cv2.THRESH_BINARY)[1]

def morph_close(img, kernel_tuple = (3, 3), reps = 2):
    """apply a series of morphological closures to image"""
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(reps):
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def sigmoid(x):
    """logistic function for pixel values in range [0, 255]"""
    return 255 / (1 + e ** (-(x - 128) / 25))
sigmoid_vec = np.vectorize(sigmoid)

def sharpen(im):
    """change pixel values to be closer to black or white"""
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_bin = sigmoid_vec(im_gray)
    im_crop = cv2.cvtColor(im_bin.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return im

def skeleton(im):
    """skeletonize image"""
    from skimage.morphology import skeletonize
    from skimage import img_as_bool

    # create binary image
    b = np.all(~img_as_bool(im), axis = 2)

    # skeletonize image
    s = skeletonize(b)

    # convert back to BGR format
    return np.where(np.repeat(np.expand_dims(s, 3), 3, 2),
                    np.full(im.shape, 0, dtype = np.uint8),
                    np.full(im.shape, 255, dtype = np.uint8))

# functions for filtering horizontal and vertical lines
filter_vert = lambda line_lst: \
              filter(lambda l: l[1] < 0.1 or l[1] > np.pi - 0.1, line_lst)
filter_horz = lambda line_lst: \
              filter(lambda l: abs(l[1] - np.pi / 2) < 0.1, line_lst)

# higher level functions

def is_line(im, line, tol = 300):
    """determine if points in line are sufficiently connected"""
    # criteria: there exists a contiguous (not separated by >5 pixels anywhere)
    # large (>25% of all pixels) segment of pixels
    # that is close to line (within 5x5 neighborhood)
    coverage_vector = [any([covers_pixel(x2, y2, im, tol)
                            for x2, y2 in iter_neighbors(x, y, 3)])
                       for x, y in iter_line(line, im)]
    num_contiguous_px = [0]
    for i in coverage_vector:
        if i == True:
            num_contiguous_px[-1] += 1
        elif num_contiguous_px[-1] != 0:
            num_contiguous_px.append(0)
    return max(num_contiguous_px) > 0.25 * len(coverage_vector)

def find_lines(im, orientation = None, is_line_check = True, tol = 100):
    """find distinct lines in image, orientation can be horiz or vert"""
    edges = cv2.Canny(im, 50, 150, apertureSize = 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)[0]

    # only keep vertical / horizontal lines, if specified
    if orientation == "horiz":
        lines = filter_horz(lines)
    elif orientation == "vert":
        lines = filter_vert(lines)
    lines = normalize_lines(lines)

    # remove lines that are outside of the bounds of the image
    def outside_image(l):
        return rho_to_xy(*l) == clamp_xy_to_border(im, *rho_to_xy(*l))
    lines = filter(lambda l: any([pt[0] > 0 for pt in rho_to_xy(*l)]), lines)
    lines = filter(lambda l: not outside_image(l), lines)

    # remove redundant lines, use multiple passes
    for _ in range(3):
        lines = consolidate_lines(lines, im, sort_by_coverage = True, tol = tol)
        lines = consolidate_lines(lines, im,
                              group_by_intersection = True,
                              sort_by_coverage = True)

    # remove non-contiguous lines, if specified
    if is_line_check:
        lines = filter(lambda l: is_line(im, l, tol = tol), lines)

    # sort lines by distance from origin
    lines = sorted(lines, key = lambda l: l[0])
    return lines

def find_horiz(im, is_line_check = True, tol = 100):
    """find distinct horizontal lines in image"""
    return find_lines(im, "horiz", is_line_check = is_line_check, tol = tol)

def find_vert(im, is_line_check = True, tol = 100):
    """find distinct vertical lines in image"""
    return find_lines(im, "vert", is_line_check = is_line_check, tol = tol)

def orient_around_table(im):
    """crop image to table, and orient so that gray header is on top"""
    (height, width, _) = im.shape

    # find horizontal / vertical lines (may be gray, may not be contiguous)
    lines_horiz = remove_distant_lines(find_horiz(im, False, tol = 500),
                                       dist_tol = height // 4)
    lines_vert = remove_distant_lines(find_vert(im, False, tol = 500),
                                      dist_tol = width // 4)

    # find border defined by lines
    getx = lambda line: line[0]; gety = lambda line: line[1]
    min_y = min(map(gety, clamp_xy_to_border(im, *rho_to_xy(*lines_horiz[0]))))
    max_y = max(map(gety, clamp_xy_to_border(im, *rho_to_xy(*lines_horiz[-1]))))
    min_x = min(map(getx, clamp_xy_to_border(im, *rho_to_xy(*lines_vert[0]))))
    max_x = max(map(getx, clamp_xy_to_border(im, *rho_to_xy(*lines_vert[-1]))))

    # add padding to border
    min_y = int(min(max(min_y - 50, 0), height))
    max_y = int(min(max(max_y + 50, 0), height))
    min_x = int(min(max(min_x - 50, 0), width))
    max_x = int(min(max(max_x + 50, 0), width))
    tab_height, tab_width = (max_y - min_y, max_x - min_x)

    # find header by determining portion of table with most non-white pixels
    def count_gray(im_seg):
        seg_px = im_seg.reshape([im_seg.shape[0] * im_seg.shape[1], 3])
        return float((seg_px.sum(axis = 1) < 250 * 3).sum()) / seg_px.shape[0]
    top_gray = count_gray(im[min_y : max_y - tab_height // 2, min_x : max_x])
    bot_gray = count_gray(im[min_y + tab_height // 2 : max_y, min_x : max_x])
    lft_gray = count_gray(im[min_y : max_y, min_x : max_x - tab_width // 2])
    rht_gray = count_gray(im[min_y : max_y, min_x + tab_width // 2 : max_x])
    num_rots = max([(top_gray, 0), (lft_gray, 1),
                    (bot_gray, 2), (rht_gray, 3)])[1]

    # crop and rotate clockwise until header is on top
    im = im[min_y : max_y, min_x : max_x]
    for _ in range(num_rots):
        im = cv2.flip(cv2.transpose(im), 1)
    return im

def segmented_line_erasure(im, lines_horiz, lines_vert):
    """divide horizontal lines into smaller segments before erasing"""
    (height, width, _) = im.shape
    #im_bw = remove_gray(im)
    for i, horiz in enumerate(lines_horiz):
        for vert_1, vert_2 in zip(lines_vert, lines_vert[1:]):
            pt1 = line_intersection_polar(horiz, vert_1)
            pt2 = line_intersection_polar(horiz, vert_2)

            # skip line if endpoints are too close together
            if dist(pt1, pt2) < 10:
                continue
            
            # define a rectangle around line segment of interest
            rect_lft = max(pt1[0] - 1, 0)
            rect_rht = min(pt2[0] + 1, width - 1)
            rect_top = max(min(pt1[1], pt2[1]) - 5, 0)
            rect_bot = min(max(pt1[1], pt2[1]) + 5, height - 1)

            # use denoised image for header
            #cur_im = im_bw if i <= 1 else im
            im_crop = crop_to_poly(im, [
                (rect_lft, rect_bot), (rect_lft, rect_top),
                (rect_rht, rect_top), (rect_rht, rect_bot)])
            im_bin = cv2.threshold(im_crop, 250, 255, cv2.THRESH_BINARY)[1]

            # toggle line endpoints until best fit obtained
            x0 = rect_lft
            x1 = rect_rht
            y_mid = (rect_top + rect_bot) // 2
            max_pixels = -1
            max_y0 = None
            max_y1 = None
            for dy0 in range(-6, 6):
                for dy1 in range(-6, 6):
                    y0 = y_mid + dy0
                    y1 = y_mid + dy1
                    num_pixels = sum([int(covers_pixel(x, y, im_bin, 250 * 3))
                                      for x, y in iter_line_xy((x0, y0),
                                                               (x1, y1), im_bin)])

                    # keep track of endpoint offsets that maximize pixels covered
                    if num_pixels > max_pixels:
                        max_pixels = num_pixels
                        max_y0 = y0
                        max_y1 = y1

            cv2.line(im, (int(x0), int(max_y0)), (int(x1), int(max_y1)), white, 4)
            
def grid_ocr(im, lines_horiz, lines_vert):
    """apply tesseract ocr to each square within grid"""
    data = []
    for i, (horiz_1, horiz_2) in enumerate(zip(lines_horiz, lines_horiz[1:])):
        data_ = []
        j = 0
        for vert_1, vert_2 in zip(lines_vert, lines_vert[1:]):
            # crop image to in between vertical and horizontal lines
            im_crop = crop_to_lines(im.copy(), horiz_1, horiz_2, vert_1, vert_2)

            # skip if area of image is too small
            if im_crop.shape[0] * im_crop.shape[1] < \
               im.shape[0] * im.shape[1] * 0.001:
                continue
            
            # make image 4x larger
            im_crop = cv2.resize(im_crop, dsize = None, fx = 4, fy = 4)

##            if i == 0:
##                # if header, then remove gray background
##                im_crop = remove_gray(im_crop)
##                im_crop = morph_close(im_crop)
            #else:
            #   im_crop = cv2.threshold(im_crop, 127, 255, cv2.THRESH_BINARY)[1]
            im_crop = crop_to_pixels(im_crop)

            # skip narrow / flat images
            if im_crop.shape[0] < 20 or im_crop.shape[1] < 20:
                continue

            # skip mostly blank images
            crop_px = im_crop.reshape([im_crop.shape[0] * im_crop.shape[1], 3])
            nfilled = (crop_px.sum(axis = 1) < 250 * 3).sum()
            if nfilled < im_crop.shape[0] * im_crop.shape[1] * 0.01:
                continue

            # sharpen image
            im_crop = sharpen(im_crop)

            # scale up image by 2x
            im_crop = cv2.resize(im_crop, dsize = None, fx = 2, fy = 2)
            
            # configure ocr to work on single digits, for certain columns
            if i != 0 and data[0][j] in single_char_headers:
                local_conf = tess_conf + " -psm 10"
            if i != 0 and data[0][j] in single_block_headers:
                local_conf = tess_conf + " -psm 6"
            else:
                local_conf = tess_conf + " -psm 4"

            # reduce penalty for mixing alphabetic + numeric chars
            #if i != 0 and data[0][j] in mixed_char_headers:
            local_conf += " " + mixed_char_config
            local_conf += " " + small_words_config

            if i != 0 and data[0][j] in non_dict_headers:
                local_conf += " " + non_dict_config
            pil_img = Image.fromarray(im_crop, 'RGB')
            text = pytesseract.image_to_string(pil_img, config = local_conf)
            # TODO - tess assumes number w/o char (18D -> 180, C84168 -> 084168)

            # use fuzzy matching for headers, b/c of noise from gray background
            if i == 0:
                text = max(all_headers, key = lambda h: fuzz.ratio(h, text))

            # ignore noisy text
            elif text.count("-") >= 3:
                text = ""
            print(text)
            pil_img.save("pil_img.png")
            show(im_crop)
            data_.append(text)
            j += 1
        data.append(data_)
    return data

def preprocess_and_ocr(im):
    (height, width, _) = im.shape
    im2 = im.copy()

    # eliminate gray background in header
    im_no_gray = remove_gray(im)
    show(im_no_gray)

    # find pure black long horizontal lines
    lines_header_horiz = find_horiz(im_no_gray, tol = 100)

    # find line that separates header from body
    sep_line = lines_header_horiz[1]
    (_, y1), (_, y2) = clamp_xy_to_border(im, *rho_to_xy(*sep_line))

    # create denoised image header
    im_head = crop_to_poly(im_no_gray, [(0, 0), (width, 0), (width, y2), (0, y1)])

    # create black and white image of image body, without header
    bw = cv2.threshold(im, 250, 255, cv2.THRESH_BINARY)[1]
    bw_body = crop_to_poly(bw, [(0, y1), (width, y2), (width, height), (0, height)])
    im_body = crop_to_poly(im, [(0, y1), (width, y2), (width, height), (0, height)])

    # create black and white composition of image body + header
    im_bw = im_head + bw_body - 255
    show(im_bw)

    # find vertical lines
    lines_vert = find_vert(im_bw)

    # find horizontal lines
    lines_horiz = find_horiz(bw_body, tol = 500)
    lines_horiz = lines_header_horiz[:2] + lines_horiz

    write_lines(im2, lines_horiz)
    write_lines(im2, lines_vert)

    im_comp = im_head + im_body - 255
    show(im_comp)
    cv2.imwrite("comp.png", im_comp)

    # erase lines from image
    write_lines(im_comp, lines_vert, color = white, thick = 4)
    show(im_comp)

    # zoom in on segments of image, before finding + erasing lines
    segmented_line_erasure(im_comp, lines_horiz, lines_vert)
    show(im_comp)

    # perform ocr on all squares in the table
    return grid_ocr(im_comp, lines_horiz, lines_vert)

#laser_dir = "C:/Users/vge2/Downloads/Laser Audits/Laser Audits/"
laser_dir = "C:/Users/vge2/Downloads/Laser Images/"
for file_name in os.listdir(laser_dir):
    file_path = laser_dir + file_name
    im = cv2.imread(file_path)

    # reduce image size by 1/4, using gaussian blur to downsample
    for _ in range(2):
        im = cv2.pyrDown(im)

    # check if image is laser report or table
    res = cv2.matchTemplate(im, report_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val > 0.4 and max_loc[1] < im.shape[0] / 5:
        continue

    im = orient_around_table(im)
    print(preprocess_and_ocr(im))

    show(im)
