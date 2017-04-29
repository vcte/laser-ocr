import os
import cv2
import time
import itertools
import numpy as np
from math import *
from random import *
from skimage.transform import hough_circle
from skimage.feature.peak import _prominent_peaks

# opencv hsv: h = [0, 180], s = [0, 255], v = [0, 255]
# opencv: red is 0, yellow is 30, green is 60, blue is 120

char = "marth"
idle = cv2.cvtColor(cv2.imread("frames/marth/idle_1/idle_1-0120.png"),
                    cv2.COLOR_BGR2HSV)
height, width, _ = idle.shape
size = min(height, width)
red_id = 0; yellow_id = 1; green_id = 2; blue_id = 3;
color_ids = [red_id, yellow_id, green_id, blue_id]
red = (0, 255, 127); yellow = (30, 255, 127)
green = (60, 255, 127); blue = (120, 255, 127);
black = (0, 0, 0); white = (0, 0, 255)
id_to_color = {red_id : red, yellow_id : yellow,
               green_id : green, blue_id : blue}

# circle := [0, c, x, y, r]
# rectangle := [1, c, x, y, w, h, a] (origin as top left)
#    w is dist from center to side, h is dist from center to semicircle center

# Hyperparameters
# min / max radius and total number of circles to return for hough transform
MIN_CIRCLE_RADIUS = 10
MAX_CIRCLE_RADIUS = 50
TOTAL_NUM_CIRCLES = 1500

# max number of 'bad' semicircles to process before skipping the rest
MAX_CONSECUTIVE_MISSES = 100

# determine which lines are similar
SIMILAR_LINE_RHO_THRESHOLD = 5
SIMILAR_LINE_THETA_THRESHOLD = 0.03

# semi/circle coverage params
CIRCLE_COVERAGE_MAX_D = 1
CIRCLE_COVERAGE_TOL = 0.95
SEMICIRCLE_COVERAGE_TOL = 0.95

# max difference between parameters before circles are considered identical
CIRCLE_PARAM_DIFF_THRESHOLD = 3

# max difference between parameters before semicircles are considered identical
SEMICIRCLE_PARAM_DIFF_THRESHOLD = 5

# lower bound on number of consecutive points a line should cover
MIN_LINE_COVERAGE = 20

# angle / distance thresholds for two lines to be considered parallel
PARALLEL_LINE_ANGLE_DIFF = 0.02
PARALLEL_LINE_MIN_DIST = 10

# max distance between semicircle endpoints and two lines for them to match
SEMI_LINE_MAX_DIST = 8

# max number of attempts to find closest pt that isn't already covered
# append_closest_pt routine is a bottleneck, so this param should be kept low
MAX_TRIES_FIND_CLOSEST_PT = 3

# min / max distance between two points before interpolating between them
MIN_INTERP_DIST = 2
MAX_INTERP_DIST = 5

# lower bound on number of pts in edge map that a set of points should cover
CLOSEST_PT_COVERAGE_TOL = 0.8

# controls which pairs of semicircles are candidates for forming capsule
SEMI_OPP_DIR_TOL = 54
MAX_SEMI_RADII_DIFF = 10
MIN_SEMI_CENTER_DIST = 100

# minimum capsule width; distance btwn two lines of capsule = 2 x capsule width
MIN_CAPSULE_WIDTH = 8
MIN_CAPSULE_HEIGHT = 2

# controls which capsules should be filtered based on number of pts it covers
CAPSULE_COVERAGE_MAX_D = 1
CAPSULE_COVERAGE_TOL = 0.75

# max difference between parameters before capsules are considered identical
CAPSULE_PARAM_DIFF_THRESHOLDS = [10, 10, 5, 5]
CAP_ANGLE_DIFF = 15

# percent of color map that new shape should occupy, relative to its own size
NEW_SHAPE_MIN_PERC_COVERAGE = 0.05

# each shape should contribute this many unique pixels, relative to its own size
SHAPE_MIN_PERC_UNIQUE = 0.01

# IO functions for HSV images
def show_hsv(im):
    cv2.imshow("image", cv2.cvtColor(np.array(im, dtype = np.uint8),
                                     cv2.COLOR_HSV2BGR))
    cv2.waitKey()

def write_hsv(filename, im):
    cv2.imwrite(filename, cv2.cvtColor(im, cv2.COLOR_HSV2BGR))

def create_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def read_poly_file(fname):
    circles = []; circle_colors = []; capsules = []; cap_colors = []
    with open(fname, mode = "r", encoding = "utf-8") as f:
        for line in f.readlines():
            col, _, poly = line.partition(", ")
            poly = tuple(map(float, poly.strip().strip("()").split(", ")))
            if len(poly) == 3:
                circles.append(poly)
                circle_colors.append(int(col))
            elif len(poly) == 5:
                capsules.append(poly)
                cap_colors.append(int(col))
            else:
                raise ValueError("unknown polygon")
    return circles, circle_colors, capsules, cap_colors

# decorator that keeps track of total amount of time used by a function
record_name_to_index = {}
timing_records = [0]
count_records = [0]
sos_records = [0]
def record_time(name, parent = "", generator = False):
    # associate name, parent w/ index
    global record_name_to_index, timing_records, count_records
    index = record_name_to_index.get((name, parent))
    if index is None:
        index = max(record_name_to_index.values() or [0]) + 1
        record_name_to_index[(name, parent)] = index
        timing_records.append(0)
        count_records.append(0)
        sos_records.append(0)

    def update_records(index, dt):
        if count_records[index] > 0:
            delta_1 = dt - timing_records[index] / count_records[index]
        else:
            delta_1 = dt
        timing_records[index] += dt
        count_records[index] += 1
        delta_2 = dt - timing_records[index] / count_records[index]
        sos_records[index] += delta_1 * delta_2
        
    def wrapper(f):
        def wrapper_(*args, **kwargs):
            t = time.time()
            ret = f(*args, **kwargs)
            update_records(index, time.time() - t)
            return ret
        return wrapper_

    def gen_wrapper(f):
        def wrapper_(*args, **kwargs):
            t = time.time()
            yield from f(*args, **kwargs)
            update_records(index, time.time() - t)
        return wrapper_
    
    return wrapper if not generator else gen_wrapper

def print_timings(parent = "", indent = "", output = print):
    for name, par in sorted(record_name_to_index.keys(),
                            key = lambda x: record_name_to_index[x]):
        if parent == par:
            index = record_name_to_index[(name, par)]
            output(indent + name + " [" + str(count_records[index]) +  "]: " + \
                   str(timing_records[index]) + " +/- " +
                   str(sqrt(sos_records[index] / (count_records[index] + 0.1))))
            print_timings(name, indent + " ", output)

# convert line from rho, theta to two points that define the line
def rho_to_xy(rho, theta, scale = 2000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    x1 = x0 + scale*(-b)
    y1 = y0 + scale*(a)
    x2 = x0 - scale*(-b)
    y2 = y0 - scale*(a)
        
    return ((int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))))

# convert line represented by two points into y = mx + b form
# if line is vertical, return (None, x-intercept)
def xy_to_mb(pt1, pt2):
    (x1, y1) = pt1
    (x2, y2) = pt2
    if abs(x1 - x2) > 0.0001:
        m = float(y1 - y2) / (float(x1 - x2))
        b = y1 - m * x1
    else:
        m, b = None, x1
    return m, b

# blend hsv images, weighted by value of the color
def blend_images(im1, im2):
    def blend(x, y, w1, w2):
        if w1 + w2 > 0:
            return int((x * w1 + y * w2) / (w1 + w2))
        else:
            return max(x, y)
    for x, y in np.ndindex(width, height):
        hue = blend(im1[y, x, 0], im2[y, x, 0], im1[y, x, 2], im2[y, x, 2])
        sat = max(im1[y, x, 1], im2[y, x, 1])
        val = blend(im1[y, x, 2], im2[y, x, 2], 1, 1)
        im1[y, x] = np.array([hue, sat, val], dtype = np.uint8)
    return im1

# calculate meta parameters of a capsule
def meta_capsule(x, y, w, h, a):
    b = 90 - a
    hs = int(round(h * sin(a * pi / 180)))
    hc = int(round(h * cos(a * pi / 180)))
    ws = int(round(w * sin(b * pi / 180)))
    wc = int(round(w * cos(b * pi / 180)))

    pts = [(x + hs - ws, y - hc - wc), (x + hs + ws, y - hc + wc),
           (x - hs + ws, y + hc + wc), (x - hs - ws, y + hc - wc)]
    return hs, hc, ws, wc, pts

# drawing routines
def draw_lines(im, lines, color = white, thick = 1):
    for line in lines:
        rho, theta = line[0]
        ((x1, y1), (x2, y2)) = rho_to_xy(rho, theta)
        cv2.line(im, (x1, y1), (x2, y2), color, thick)

def draw_semicircle(im, x, y, r, a, color = white, thickness = -1):
    cv2.ellipse(im, (int(x), int(y)), (int(r), int(r)),
                a, 180, 360, color, thickness)

def draw_capsule(im, x, y, w, h, a, color, thickness = -1):
    x = int(x); y = int(y);
    hs, hc, ws, wc, pts = meta_capsule(x, y, w, h, a)

    # draw the rotated rectangle
    if thickness != -1:
        # draw outline
        cv2.line(im, pts[1], pts[2], color, thickness)
        cv2.line(im, pts[0], pts[3], color, thickness)
    else:
        # draw a filled rectangle
        pts = np.array([pts], dtype = np.int32)
        cv2.fillPoly(im, pts, color)

    # draw circles on both ends of rectangle
    draw_semicircle(im, x + hs, y - hc, w, a, color, thickness)
    draw_semicircle(im, x - hs, y + hc, w, (a + 180) % 360, color, thickness)

def draw_shapes(shapes):
    im = np.zeros((height, width, 3), np.uint8)
    for shape in shapes:
        color = id_to_color[shape[1]]
        im2 = np.zeros((height, width, 3), np.uint8)
        if shape[0] == 0:       # draw circle
            cv2.circle(im2, (shape[2], shape[3]), shape[4], color, -1)
        else:                   # draw rectangle
            cv2.rectangle(im2, [shape[2], shape[3]],
                          [shape[2] + shape[4], shape[3] + shape[5]], color, -1)
        im = blend_images(im, im2)
    return im

def draw_shapes_fast(shapes):
    im = np.zeros((height, width, 3), np.uint8)
    for shape in sorted(shapes, key = lambda s: s[1]):
        # shapes are sorted so that colors w/ higher hue painted over lower hues
        color = id_to_color[shape[1]]
        if shape[0] == 0:
            cv2.circle(im, (shape[2], shape[3]), shape[4], color, -1)
        else:
            im = draw_capsule(im, *shape[2:7], color)
    return im

# modify image so that it is made up of 5 colors
record_time("quantize", "preprocess")
def quantize_image(im):
    print("   quantizing the image")
    # indistinct colors + black, typically on border of hitbox
    def indistinct(px):
        return px[1] < 127 or px[2] < 64
    
    # determine which colors are present in image
    col_count = dict([(col, 0) for col in color_ids])
    for y, x, _ in zip(*np.nonzero(im == 255)):
        if indistinct(im[y, x]):
            continue

        # match the hue against the hue of all pure colors
        pix_col = None
        for col in color_ids:
            if abs(id_to_color[col][0] - im[y, x, 0]) <= 3:
                pix_col = col
                break
        if pix_col != None:
            col_count[pix_col] += 1
    colors = [id_to_color[col] for col in col_count if col_count[col] >= 25]

    # determine hues and values of possible color combinations
    base_hues = [h for (h, s, v) in colors]
    all_hues = list(set([h for (h, v) in
                         construct_color_palette(base_hues, max_colors = 4)]))
    all_vals = list(set([v for (h, v) in
                         construct_color_palette(base_hues, max_colors = 6)]))

    # quantize image
    im2 = np.zeros((height, width, 3), np.uint8)
    wrap_cmp = lambda a, b: min((a - b) % 180, (b - a) % 180)
    if len(base_hues) > 0:
        for y, x, _ in zip(*np.nonzero(im > 127)):
            if im2[y, x, 1] == 0 and not indistinct(im[y, x]):
                val = min(all_vals, key = lambda v: abs(v - im[y, x, 2]))
                hue = min(all_hues, key = lambda h: wrap_cmp(h, im[y, x, 0]))
                im2[y, x] = np.array([hue, 255, val], dtype = np.uint8)
    return im2

# find edges in an image
record_time("edge_map", "preprocess")
def edge_map(im):
    print("   computing edge map")

    # determine if each pixel color is equal to adjacent pixel color
    # for each cardinal direction [up, down, left, right]
    shifts = [(np.roll(im, shift, axis) != im).any(axis = 2)
              for axis in [0, 1] for shift in [-1, 1]]

    # determine if pixel should be an edge, for each pixel
    cond = np.logical_or.reduce(shifts)

    # fill in edge map so that edges are white and non-edges are black
    edges_ = np.zeros((height, width, 3), np.uint8)
    edges_[:, :, 2] = cond.astype(np.uint8) * 255
    return edges_

# thin edge map so that lines of width 2 becomes lines of width 1
record_time("thin_edges", "preprocess")
def thin_edges(edges):
    # upscale by 2x
    upscaled = cv2.resize(edges, dsize = None, fx = 2, fy = 2)

    # apply erosion to upscaled edge map
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thin = cv2.erode(upscaled, cross)

    # binarize new edge map
    thin = np.where(thin > 255 / 3,
                    np.zeros(thin.shape, dtype = np.uint8) + 255,
                    np.zeros(thin.shape, dtype = np.uint8))

    # scale back to original size
    return cv2.resize(thin, dsize = None, fx = 0.5, fy = 0.5,
                      interpolation = cv2.INTER_NEAREST)

# calculate entropy, or 'impurity', of list of numbers
def entropy(ps):
    # ensure that list adds up to 1
    ps = [p / sum(ps) for p in ps]

    return -sum([p * log(p) for p in ps if p != 0])

# maps (h, 255, v) colors to mixture of base colors
record_time("construct_palette", "quantize")
def construct_color_palette(hues = [0, 30, 60, 120], max_colors = 6,
                            invert = False):
    # whenever color is added to mixture, new value = v + (v_max - v) / 2
    n_colors_to_val = {}
    v = 0
    for i in range(0, max_colors + 1):
        n_colors_to_val[i] = v
        v = v + (255 - v) / 2

    palette = {}
    color_range = list(range(0, max_colors + 1))
    for num_of_each_color in itertools.product(*([color_range] * len(hues))):
        total_num_colors = sum(num_of_each_color)
        if total_num_colors < 1 or total_num_colors > max_colors:
            continue

        v = n_colors_to_val[total_num_colors]
        h = sum([hue * n_colors_to_val[num]
                 for hue, num in zip(hues, num_of_each_color)]) // v

        # determine the direction of the mapping
        if not invert:
            key = (int(h), int(v)); val = num_of_each_color
        else:
            key = num_of_each_color; val = (int(h), int(v))

        # if more than 1 possible mixtures, pick the one with least impurity
        if key in palette:
            if entropy(val) < entropy(palette[key]):
                palette[key] = val
        else:
            palette[key] = val
    return palette
palette = construct_color_palette()

# use kdtree to find nearest hue, val
from sklearn.neighbors import KDTree
palette_keys = np.array(list(palette.keys()))
palette_kdt = KDTree(palette_keys, metric='euclidean')

# decompose image into a mixture of color maps
@record_time("decompose") # time: slightly less than 1 second / call
def decompose_colors(im, palette):
    # base color maps: [red_map, yellow_map, green_map, blue_map]
    n = 4
    color_maps = [np.zeros((height, width), dtype = np.uint8) for _ in range(n)]

    # iterate through all fully saturated pixels
    for y, x, _ in zip(*np.nonzero(im >= 200)):
        h, _, v = im[y][x]
        ind = palette_kdt.query([[h, v]], k=1)[1][0][0]
        num_of_each_color = palette[tuple(palette_keys[ind])]
        for i in range(n):
            color_maps[i][y][x] = num_of_each_color[i]

    # zero out maps for outlier colors
    for i in range(n):
        if color_maps[i].sum() < 100:
            color_maps[i] *= 0
    return color_maps

inv_palette = construct_color_palette(invert = True)

def col_to_index(maps):
    return maps[0] + 7 * maps[1] + 7 * 7 * maps[2] + 7 * 7 * 7 * maps[3]

# build a mapping from color index to hsv triple
transdict = dict([(col_to_index(num), (h, 255, v))
                  for (num, (h, v)) in inv_palette.items()])
trans_index = np.zeros((7 ** 4 + 1, 3), dtype = np.uint8)
for k, v in transdict.items():
    trans_index[k] = v

# reconstruct image from color maps
@record_time("recompose")
def recompose_image(color_maps):
    color_maps = [np.clip(col_map, 0, 6) for col_map in color_maps]
    index_map = col_to_index(color_maps)
    return trans_index[index_map.reshape(480 * 584)].reshape((480, 584, 3))

# helper function to iterate over neighborhood
def neighbors(max_d = 1):
    for dx in range(-max_d, max_d + 1):
        for dy in range(-max_d, max_d + 1):
            yield dx, dy

def neighbors_border(max_d = 1):
    # bottom / top
    for dx in range(-max_d, max_d + 1):
        yield dx, -max_d
        yield dx, +max_d
    # left / right
    for dy in range(-max_d + 1, max_d):
        yield -max_d, dy
        yield +max_d, dy

# squared euclidean distance between two pts
def dist2(pt1, pt2):
    return (pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + \
           (pt1[1] - pt2[1]) * (pt1[1] - pt2[1])

# euclidean distance between two pts
def dist(pt1, pt2):
    return sqrt(dist2(pt1, pt2))

# calculate info about a line, for future line-distance calculations
def dist_line_preprocess(x0, y0, x1, y1):
    m = (y1 - y0) / (x1 - x0 + 0.000001)
    a = m
    b = -1
    c = y1 - m * x1
    d = sqrt(a * a + b * b)
    return (a, b, c, d)

# calculate distance between pt and line
# assumes line has been preprocessed already
def dist_pt_line(x, y, line_info):
    (a, b, c, d) = line_info
    return abs(a * x + b * y + c) / d

# iterate over points in a line between two points
def iter_line(pt1, pt2):
    x0, y0 = int(round(pt1[0])), int(round(pt1[1]))
    x1, y1 = int(round(pt2[0])), int(round(pt2[1]))
    rev = reversed
    rev2 = lambda x: x
    if abs(y1 - y0) <= abs(x1 - x0):
        x0, y0, x1, y1 = y0, x0, y1, x1
        rev = lambda x: x
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
        rev2 = reversed
    leny = abs(y1 - y0)
    dx = x1 - x0
    sgn = 1 if y1 > y0 else -1
    if leny > 0:
        if rev == reversed:
            for i in rev2(range(leny + 1)):
                yield [i * dx // leny + x0, sgn * i + y0]
        else:
            for i in rev2(range(leny + 1)):
                yield [sgn * i + y0, i * dx // leny + x0]

# convert lines to slope, intercept form before finding intersection
# find (x, y) point intersection of two lines
# return all if lines are the same, None if lines are parallel
def line_intersection(line1, line2):
    m1, b1 = xy_to_mb(*line1); m2, b2 = xy_to_mb(*line2)
    if m1 is None and m2 is None:
        return all if abs(b1 - b2) < 0.001 else None
    if m1 is None:
        return b1, m2 * b1 + b2
    elif m2 is None:
        return b2, m1 * b2 + b1
    elif abs(m1 - m2) < 0.001:
        return all if abs(b1 - b2) < 0.001 else None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

# iterate over the coordinates of border of a circle centered at (x0, y0)
# also yields coordinates at right angle to current location
def iter_circle(x0, y0, r, octant_offset = 0):
    # ref: https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
    x = r
    y = 0
    e = 0

    xs = []
    ys = []
    while (x >= y):
        xs.append(x)
        ys.append(y)
        
        if (e <= 0):
            y += 1
            e += 2 * y + 1
        if (e > 0):
            x -= 1
            e -= 2 * x + 1

    # iterate over coordinates
    nxs = [-x for x in xs]; nys = [-y for y in ys]
    orientation = [(0, 1, False), (1, 0, True), (3, 0, False), (2, 1, True),
                   (2, 3, False), (3, 2, True), (1, 2, False), (0, 3, True)]
    orientation = orientation[octant_offset:] + orientation[:octant_offset]
    for ornt in orientation:
        rev = reversed if ornt[2] else lambda x: x
        for cs in rev(list(zip(xs, ys, nxs, nys))):
            yield (x0 + cs[ornt[0]], y0 + cs[ornt[1]])

# iterate over coordinates of border of semicircle
@record_time("iter_semi")
def iter_semi(cx, cy, cr, ca):
    pts = list(iter_circle(cx, cy, cr))
    off = int(round(((ca % 360) - 180) / 360 * len(pts)))
    sem = len(pts) // 2
    rng = pts[off : off + sem] if off >= 0 else pts[off :] + pts[: off + sem]
    for (x, y) in rng:
        yield x, y

# iterate over coordinates of border of capsule
def iter_capsule(x, y, w, h, a):
    hs, hc, ws, wc, pts = meta_capsule(x, y, w, h, a)
    yield from iter_line(pts[1], pts[2])
    yield from iter_semi(x + hs, y - hc, w, a)
    yield from iter_line(pts[0], pts[3])
    yield from iter_semi(x - hs, y + hc, w, (a + 180) % 360)

# find the first nonzero point, going in a straight line from pt1 to pt2
def first_pt(im, pt1, pt2, max_d = 0):
    # https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python
    def in_image(x, y):
        h, w, _ = im.shape
        return any([0 <= x + dx < w and 0 <= y + dy < h and
                    (im[int(y + dy)][int(x + dx)] != black).any()
                    for dx, dy in neighbors(max_d)])
    if max_d > 0:
        def in_set(x, y):
            # TODO: lazy iteration to avoid computing full list
            return any([(x + dx, y + dy) in im
                        for dx, dy in neighbors(max_d)])
    else:
        def in_set(x, y):
            return (x, y) in im
    test = in_set if type(im) == set else in_image
    for x, y in iter_line(pt1, pt2):
        if test(x, y):
            return x, y
    return None

@record_time("first_pts", "param_capsule")
def first_pts_from_center_at_angle(x, y, angle, pts):
    for da in [0, -2, 2, -5, 5]:
        dx = size * sin((angle + da) * pi / 180)
        dy = size * cos((angle + da) * pi / 180)
        pt0 = first_pt(set(pts), (x, y), (x + dx, y - dy))
        pt1 = first_pt(set(pts), (x, y), (x - dx, y + dy))
        if pt0 != None and pt1 != None:
            break
    return pt0, pt1

def choose_pt(x, y, pt0, pt1):
    if pt0 is None and pt1 is None:
        return (x, y)
    if pt0 is None:
        return pt1
    elif pt1 is None:
        return pt0
    else:
        return min((pt0, pt1), key = lambda p: dist(p, (x, y)))

@record_time("pt_in_ellipse", "lines")
def pt_in_ellipse(pt, ellipse):
    x, y = pt
    ex, ey, rx, ry, ea = ellipse
    tx = (((x - ex) * cos(ea) + (y - ey) * sin(ea)) ** 2) / rx / rx
    ty = (((x - ex) * sin(ea) - (y - ey) * cos(ea)) ** 2) / ry / ry
    return tx + ty <= 1
    
# attempt to find circle parameters that matches data points
def param_circle(xs, ys):
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    cr = int((np.sqrt(np.square(xs - cx) + np.square(ys - cy))).sum() / len(xs))
    return cx, cy, cr

# attempt to find capsule parameters that matches data points
def param_capsule_old(xs, ys):
    pts = set([(x, y) for x, y in zip(xs, ys)])
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    center = (cx, cy)
    mr = max(max(xs) - cx, cx - min(xs), max(ys) - cy, cy - min(ys))
    max_diff = -1; max_tip = -1; max_side = -1; max_major = None
    for major_pt, minor_pt in zip(iter_circle(cx, cy, mr, 0),
                                  iter_circle(cx, cy, mr, 2)):
        tip_pt = first_pt(pts, major_pt, center)
        if tip_pt != None:
            side_pt = first_pt(pts, center, minor_pt)
            if side_pt != None:
                tip_dist = dist2(tip_pt, center)
                side_dist = dist2(side_pt, center)
                diff = tip_dist - side_dist
                if diff > max_diff:
                    max_diff = diff
                    max_tip  = tip_dist
                    max_side = side_dist
                    max_major = major_pt
    angle = atan2(max_major[0] - cx, max_major[1] - cy)
    w = sqrt(max_side)
    h = sqrt(max_tip) - w
    return cx, cy, w, h, angle

# find parameters of capsule that best matches the given points
@record_time("param_capsule", "capsules")
def param_capsule(pts):
    xs, ys = zip(*pts)
    cx = int(round(sum(xs) / len(xs)))
    cy = int(round(sum(ys) / len(ys)))

    # iteratively improve center / width / height / angle estimates
    for _ in range(3):
        # find directions of most / least variance
        pts_ = [(x - cx, y - cy) for x, y in pts]
        u, s, v = np.linalg.svd(pts_)

        # determine angles of major / minor axis from vertical line
        major_angle = atan2(v[0][1], v[0][0]) * 180 / pi + 90
        minor_angle = atan2(v[1][1], v[1][0]) * 180 / pi + 90
        
        # determine distance of farthest / shortest pt from center
        major_pts = first_pts_from_center_at_angle(cx, cy, major_angle, pts)
        major_dist = dist(choose_pt(cx, cy, *major_pts), (cx, cy))
        minor_pts = first_pts_from_center_at_angle(cx, cy, minor_angle, pts)
        minor_dist = dist(choose_pt(cx, cy, *minor_pts), (cx, cy))

        # apply corrective shift if distances are dissimilar
        if major_pts[0] != None and major_pts[1] != None:
            major_dist_0 = dist(major_pts[0], (cx, cy))
            major_dist_1 = dist(major_pts[1], (cx, cy))
            if abs(major_dist_0 - major_dist_1) > 1:
                major_dist = (major_dist_1 + major_dist_0) / 2
                dx = sin(major_angle * pi / 180)
                dy = cos(major_angle * pi / 180)
                dd = major_dist - major_dist_1
                cx += dx * dd; cy -= dy * dd
        if minor_pts[0] != None and minor_pts[1] != None:
            minor_dist_0 = dist(minor_pts[0], (cx, cy))
            minor_dist_1 = dist(minor_pts[1], (cx, cy))
            if abs(minor_dist_0 - minor_dist_1) > 1:
                minor_dist = (minor_dist_1 + minor_dist_0) / 2
                dx = sin(minor_angle * pi / 180)
                dy = cos(minor_angle * pi / 180)
                dd = minor_dist - minor_dist_1
                cx += dx * dd; cy -= dy * dd
    
    return (cx, cy, minor_dist, major_dist - minor_dist, major_angle)

# compute intersection over union, works for black and white hsv images
def iou_fast(im1, im2):
    # determine how many non-black pixels are in both images in total
    u = (im1.astype(np.int32) + im2.astype(np.int32)) > 0

    # determine how many non-black channels in both images match
    i = np.bitwise_and(abs(im1.astype(np.int32) - im2.astype(np.int32)) <= 3, u)
    
    return i.sum() / u.sum()

def semicircles_identical(semicircle_1, semicircle_2):
    return all([abs(a - b) < SEMICIRCLE_PARAM_DIFF_THRESHOLD
                for a, b in zip(semicircle_1, semicircle_2)])

def calc_circle_covered(edges_gray, cx, cy, cr, max_d = CIRCLE_COVERAGE_MAX_D):
    return [any([0 <= x + dx < width and 0 <= y + dy < height and
                 edges_gray[int(round(y + dy))][int(round(x + dx))] > 0
                 for dx, dy in neighbors(max_d)])
            for (x, y) in iter_circle(cx, cy, cr)]

@record_time("circles", "hough")
def find_circles(edges, edges_gray):
    # apply hough transform with lax settings, to find large number of circles
    print("    finding semi/circles")
    hough_radii = np.arange(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS)
    hough_res = hough_circle(edges_gray, hough_radii)

    # extract ~10 circles for each radius
    # this ensures that the algorithm will produce a diverse sample
    # instead of picking small circles with dense coverage
    n_circles = TOTAL_NUM_CIRCLES // (MAX_CIRCLE_RADIUS - MIN_CIRCLE_RADIUS)
    hough_x = []; hough_y = []; hough_r = []; accums = []
    for radius, h in zip(hough_radii, hough_res):
        h_p, x_p, y_p = _prominent_peaks(h, num_peaks = n_circles)
        hough_x.extend(x_p)
        hough_y.extend(y_p)
        hough_r.extend([radius, ] * len(h_p))
        accums.extend(h_p)

    # determine which circles / semicircles match the edge map
    circles = []
    circles_coverage = []
    semicircles = []
    semicircles_coverage = []
    num_consecutive_misses = 0
    for cx, cy, cr, _ in sorted(zip(hough_x, hough_y, hough_r, accums),
                                key = lambda h: h[3], reverse = True):
        # skip rest of circles, if circles are starting to get 'bad'
        if num_consecutive_misses >= MAX_CONSECUTIVE_MISSES:
            break

        # determine how many pixels the circle covers in the edge map
        covered = calc_circle_covered(edges_gray, cx, cy, cr)

        # detect circle
        circle_percent_coverage = sum(covered) / len(covered)
        if circle_percent_coverage > CIRCLE_COVERAGE_TOL:
            circles.append((cx, cy, cr))
            circles_coverage.append(circle_percent_coverage)
            continue

        # check if impossible to form semicircle
        elif circle_percent_coverage < SEMICIRCLE_COVERAGE_TOL / 2:
            num_consecutive_misses += 1
            continue

        # detect semicircle
        len_semi = len(covered) // 2
        n_current = sum(covered[: len_semi])
        n_max = -1; offset_max = -1
        for i in range(len(covered)):
            if n_current > n_max:
                n_max = n_current
                offset_max = i
            n_current -= int(covered[i])
            n_current += int(covered[(i + len_semi) % len(covered)])

        if n_max / len_semi > SEMICIRCLE_COVERAGE_TOL:
            angle = offset_max / len(covered) * 360 + 180

            # don't add semicircle if it is nearly identical to an existing semi
            if not any([semicircles_identical(sem, (cx, cy, cr, angle))
                        and n_max / len_semi <= cov
                        for sem, cov
                        in zip(semicircles, semicircles_coverage)]):
                semicircles.append((cx, cy, cr, angle))
                semicircles_coverage.append(n_max / len_semi)
            num_consecutive_misses = 0
        else:
            num_consecutive_misses += 1
    return circles, circles_coverage, semicircles, semicircles_coverage

@record_time("lines", "hough")
def find_lines(edges_gray):
    # apply hough transform with lax settings, to find large number of lines
    print("    finding lines")
    hough_lines = cv2.HoughLines(edges_gray, 1, np.pi/180, MIN_LINE_COVERAGE)
    if hough_lines is None:
        return []

    # remove lines that are too similar
    def any_similar_lines(line, lines):
        for line2 in lines:
            if abs(line[0][0] - line2[0][0]) < SIMILAR_LINE_RHO_THRESHOLD and \
               abs(line[0][1] - line2[0][1]) < SIMILAR_LINE_THETA_THRESHOLD:
                return True
        return False
    lines = []
    for line in hough_lines:
        if not any_similar_lines(line, lines):
            lines.append(line)

    # filter out lines that do not seem to lie on consecutive line of pixels
    bounds = image_bounds(edges_gray)
    line_covs = [(line, max(list(
        consecutive_pixels_covered(edges_gray, line[0], bounds)) or [(-1, 0)]))
                 for line in lines]
    line_covs = [(l, (cov, seg)) for l, (cov, seg) in line_covs
                 if cov >= MIN_LINE_COVERAGE]

    # sort by the size of the line segment that the line covers
    line_covs = sorted(line_covs, key = lambda x: x[1][0], reverse = True)

    # filter out lines whose segment is contained within another line's segment
    lines = []
    seg_ellipses = []
    for line, _ in line_covs:
        cov, (first_pt, last_pt) = max(consecutive_pixels_covered(
            edges_gray, line[0], bounds, max_d = 0))
        if first_pt != None and last_pt != None and \
           not any([pt_in_ellipse(first_pt, ellipse) and
                    pt_in_ellipse(last_pt, ellipse)
                    for ellipse in seg_ellipses]):
            lines.append(line)

            # parameters of ellipse closely surrounding the line segment
            ex = (first_pt[0] + last_pt[0]) / 2
            ey = (first_pt[1] + last_pt[1]) / 2
            rx = dist(first_pt, (ex, ey)) + 3
            ry = 3
            ea = atan2(last_pt[1] - ey, last_pt[0] - ex)
            seg_ellipses.append((ex, ey, rx, ry, ea))
    return lines

def nearly_parallel(line, line2):
    diff_theta = abs(line[0][1] - line2[0][1])
    return diff_theta < pi * PARALLEL_LINE_ANGLE_DIFF or \
           diff_theta > pi * (1 - PARALLEL_LINE_ANGLE_DIFF)

@record_time("semi_touches", "capsules")
def semi_touches_both_lines(end_pt_0, end_pt_1, info, info2):
    pt_is_close_to_line = [[dist_pt_line(*pt, line_info) < SEMI_LINE_MAX_DIST
                            for line_info in [info, info2]]
                           for pt in [end_pt_0, end_pt_1]]
    return (pt_is_close_to_line[0][0] and pt_is_close_to_line[1][1]) or \
           (pt_is_close_to_line[0][1] and pt_is_close_to_line[1][0])

@record_time("append_pt", "append_pts")
def append_closest_pt_or_neighbor(cache, set, x, y):
    x = int(round(x)); y = int(round(y))
    if 0 <= x < width and 0 <= y < height:
        for tries in range(MAX_TRIES_FIND_CLOSEST_PT):
            x2, y2 = cache[y][x][tries]
            if x2 != -1 or y2 != -1:
                pt = (x2, y2)
                if pt not in set:
                    set.add(pt)
                    return pt
            else:
                break

@record_time("append_pts", "capsules")
def append_pts_and_report_coverage(im, cache, set, itr):
    count = 0; total = 0
    last_pt = (-width, -height)
    for x, y in itr:
        pt = append_closest_pt_or_neighbor(cache, set, x, y)
        if pt != None:
            count += 1

            # interpolate if possible / reasonably close
            if MIN_INTERP_DIST <= dist(pt, last_pt) <= MAX_INTERP_DIST:
                for x, y in iter_line(pt, last_pt):
                    if im[y][x] > 0 and (x, y) not in set:
                        set.add((x, y))
            last_pt = pt
        total += 1
    return count / total > CLOSEST_PT_COVERAGE_TOL if total > 0 else 0

@record_time("consecutive_pixels", "lines", generator = True)
def consecutive_pixels_covered(im, line, bounds = None, max_d = 1):
    if bounds != None:
        min_x = max(bounds[0] - max_d - 1, max_d)
        max_x = min(bounds[1] + max_d + 1, width - max_d)
        min_y = max(bounds[2] - max_d - 1, max_d)
        max_y = min(bounds[3] + max_d + 1, height - max_d)
    else:
        min_x, max_x, min_y, max_y = max_d, width - max_d, max_d, height - max_d
    pt1, pt2 = rho_to_xy(*line, scale = size)
    first_pt = None
    coverage = 0
    for x, y in iter_line(pt1, pt2):
        if min_x <= x < max_x and min_y <= y < max_y:
            if any([im[y + dy][x + dx] > 0
                for dx, dy in neighbors(max_d)]):
                if first_pt is None:
                    first_pt = (x, y)
                coverage += 1
            else:
                first_pt = None
                coverage = 0
            last_pt = (x, y)
            yield coverage, (first_pt, last_pt)

def pts_in_neighborhood_covered(im, x, y, max_d):
    for dx, dy in neighbors(max_d):
        x2 = int(round(x + dx)); y2 = int(round(y + dy))
        yield 0 <= x2 < width and 0 <= y2 < height and im[y2][x2] > 0

def pts_in_neighborhood_covered_fast(im, x, y, max_d):
    for dx, dy in neighbors(max_d):
        yield im[int(round(y + dy))][int(round(x + dx))] > 0
    
@record_time("calc_coverage")
def calc_capsule_coverage(im, capsule, max_d = 1):
    # use an optimized version of subroutine, if capsule lies entirely in image
    x, y, w, h, _ = capsule
    if 15 < x - abs(w) - abs(h) and x + abs(w) + abs(h) < width - 15 and \
       15 < y - abs(w) - abs(h) and y + abs(w) + abs(h) < height - 15:
        coverage_test = pts_in_neighborhood_covered_fast
    else:
        coverage_test = pts_in_neighborhood_covered

    coverage = [any(coverage_test(im, x, y, max_d))
                for x, y in iter_capsule(*capsule)]
    return sum(coverage) / len(coverage)

# find boundaries of rectangle covering all non-black pixels in image
# input: im = single-channel grayscale image
# assumes that image has at least one non-black pixel
def image_bounds(im):
    height, width = im.shape
    min_x = width - 1; max_x = 0; min_y = height - 1; max_y = 0
    for x in range(width):
        if im[:,x].sum() > 0:
            min_x = x
            break
    for x in range(width - 1, 0, -1):
        if im[:,x].sum() > 0:
            max_x = x
            break
    for y in range(height):
        if im[y,:].sum() > 0:
            min_y = y
            break
    for y in range(height - 1, 0, -1):
        if im[y,:].sum() > 0:
            max_y = y
            break
    return min_x, max_x, min_y, max_y

def find_closest_line(line_pts):
    ls = [(min([dist2(pt, line_pts[0])
                for pt in iter_line(*rho_to_xy(*line[0], scale = size))]) +\
           min([dist2(pt, line_pts[1])
                for pt in iter_line(*rho_to_xy(*line[0], scale = size))]),
           line)
          for line in lines]
    if len(ls) > 0:
        return min(ls, key = lambda l: l[0])[1]
    else:
        print("no lines close to: " + str(line_pts))

def find_closest_semi(semi_param):
    ss = [(sum([abs(x - y) for x, y in zip(semi[:3], semi_param[:3])]) +\
           abs((semi[3] - semi_param[3] + 45) % 360 - 45),
          semi)
          for semi in semicircles]
    if len(ss) > 0:
        return min(ss, key = lambda l: l[0])[1]
    else:
        print("no circles close to: " + str(semi_param))

def debug_capsules(edges, edges_gray, circles, semicircles, lines,
                   line_pts_1, line_pts_2, semi_param_1, semi_param_2):
    line_1 = find_closest_line(line_pts_1)
    line_2 = find_closest_line(line_pts_2)
    lines_ = [line_1, line_2]
    semi_1 = find_closest_semi(semi_param_1)
    semi_2 = find_closest_semi(semi_param_2)
    semis_ = [semi_1, semi_2]
    edges_ = edges.copy()
    for line in lines_:
        out = cv2.line(edges_, *rho_to_xy(*line[0]), blue, 1)
    for (cx, cy, cr, ca) in semis_:
        out = cv2.ellipse(edges_, (cx, cy), (cr, cr), ca, 180, 360, red, 1)
    print(lines_); print(semis_)
    show_hsv(edges_)
    return find_capsules(edges, edges_gray, circles, semis_, lines_, False)

@record_time("capsules", "hough")
def find_capsules(edges, edges_gray, circles, semicircles, lines, quiet = True):
    print("    finding capsules")

    # cache the coordinates of endpoints + top of semicircles
    end_pts = []
    for cx, cy, cr, ca in semicircles:
        dx = cr * cos(ca * pi / 180); dy = cr * sin(ca * pi / 180)
        end_pts.append([(cx + dx, cy + dy), (cx - dx, cy - dy),
                        (cx + dy, cy - dx)])

    # cache the results of preprocessing each line
    line_infos = []
    for line in lines:
        (x0, y0), (x1, y1) = rho_to_xy(*line[0])
        line_infos.append(dist_line_preprocess(x0, y0, x1, y1))

    min_x, max_x, min_y, max_y = image_bounds(edges_gray)

    # cache the closest k points for each point
    @record_time("fill_cache_closest", "capsules")
    def fill_cache_closest(x, y):
        if min_x - 3 <= x <= max_x + 3 and min_y - 3 <= y <= max_y + 3:
            tries = 0
            if edges_gray[y][x] > 0:
                cache_closest[y, x, tries] = np.array((x, y), dtype = np.int32)
                tries += 1
            for max_d in range(1, 3):
                for dx, dy in neighbors_border(max_d):
                    if 0 <= x + dx < width and 0 <= y + dy < height and \
                       edges_gray[y + dy][x + dx] > 0:
                        cache_closest[y, x, tries] = np.array((x + dx, y + dy),
                                                              dtype = np.int32)
                        tries += 1
                        if tries >= MAX_TRIES_FIND_CLOSEST_PT:
                            return
    
    cache_closest = np.zeros((height, width, MAX_TRIES_FIND_CLOSEST_PT, 2),
                             dtype = np.int32) - 1
    for x, y in np.ndindex(width, height):
        fill_cache_closest(x, y)

    # determine which pairs of lines can potentially be part of a capsule
    # criteria: must be parallel, but not too close to one another
    line_pairs = [((line, info), (line2, info2))
                  for i, (line, info) in enumerate(zip(lines, line_infos))
                  for line2, info2 in zip(lines[i + 1 :], line_infos[i + 1 :])
                  if nearly_parallel(line, line2)
                  and abs(line[0][0] - line2[0][0]) > PARALLEL_LINE_MIN_DIST]
    quiet or print("number of line pairs: " + str(len(line_pairs)))

    # determine which pairs of lines and semicircles make up a capsule
    capsules = []
    capsules_coverage = []
    
    # find semicircles whose endpoints are close to both lines
    for j, (semi, s1_end_pts) in enumerate(zip(semicircles, end_pts)):
        for semi2, s2_end_pts in zip(semicircles[j + 1 :], end_pts[j + 1 :]):
            if abs(abs(semi[3] - semi2[3]) - 180) % 360 > SEMI_OPP_DIR_TOL:
                quiet or print("semicircles are not in opposite dirs.")
                continue
            elif abs(semi[2] - semi2[2]) > MAX_SEMI_RADII_DIFF:
                quiet or print("semicircle radii are too different")
                continue
            center_distance = dist2(semi[:2], semi2[:2])
            if center_distance < MIN_SEMI_CENTER_DIST:
                quiet or print("semicircles are too close together")
                continue
            elif center_distance > dist2(s1_end_pts[2], s2_end_pts[2]):
                quiet or print("semicircles not facing away")
                continue
            
            for ((line, info), (line2, info2)) in line_pairs:
                if not semi_touches_both_lines(*s1_end_pts[:2], info, info2):
                    quiet or print("semicircle 1 not close to lines")
                    continue
                elif not semi_touches_both_lines(*s2_end_pts[:2], info, info2):
                    quiet or print("semicircle 2 not close to lines")
                    continue
                
                # find closest pts from lines of capsule
                capsule_pix = set()
                covered = True
                for l in [line, line2]:
                    i_1 = line_intersection(rho_to_xy(*l[0]), s1_end_pts[:2])
                    i_2 = line_intersection(rho_to_xy(*l[0]), s2_end_pts[:2])
                    if i_1 != None and i_2 != None and \
                       i_1 != all  and i_2 != all:
                        covered &= append_pts_and_report_coverage(
                            edges_gray, cache_closest,
                            capsule_pix, iter_line(i_1, i_2))
                    else:
                        covered = False
                if not covered:
                    quiet or print("lines do not cover sufficient points")
                    continue

                # find closest pts from semicircles of capsule
                for (cx, cy, cr, ca) in [semi, semi2]:
                    covered &= append_pts_and_report_coverage(edges_gray, 
                        cache_closest, capsule_pix,
                        iter_semi(cx, cy, cr, ca))
                if not covered:
                    quiet or print("semicircles don't cover enough pts")
                    continue

                # determine parameters of capsule
                capsule = (_, _, w, h, a) = param_capsule(capsule_pix)

                if not quiet:
                    print(capsule)
                    edges_ = edges.copy()
                    out = draw_capsule(edges_, *capsule, green, 1)
                    for x, y in capsule_pix:
                        edges_[y][x] = np.array(yellow, dtype = np.uint8)
                    write_hsv("debug_cap.png", edges_)
                    show_hsv(edges_)

                # skip if capsule is too small
                if w < MIN_CAPSULE_WIDTH: #or h < 3:
                    quiet or print("capsule is too small: " + str((w, h)))
                    continue

                # calculate what percentage of capsule covers points
                coverage = calc_capsule_coverage(edges_gray, capsule,
                                                 CAPSULE_COVERAGE_MAX_D)
                quiet or print("% coverage: " + str(coverage))

                # if capsule covers enough points, then add it to list
                if coverage > CAPSULE_COVERAGE_TOL:
                    capsules.append(capsule)
                    capsules_coverage.append(coverage)

    # if multiple capsules are similar, pick the one w/ highest pixel coverage
    capsules_unique = []
    coverage_unique = []
    tols = CAPSULE_PARAM_DIFF_THRESHOLDS
    for cap, cov in sorted(zip(capsules, capsules_coverage),
                           key = lambda c: c[1], reverse = True):
        if not any([all([abs(param_1 - param_2) < tol
                         for param_1, param_2, tol in zip(cap, cap2, tols)])
                    and abs((cap[4] - cap2[4] + 15) % 180 - 15) < CAP_ANGLE_DIFF
                    for cap2 in capsules_unique]):
            capsules_unique.append(cap)
            coverage_unique.append(cov)
    
    return capsules_unique, coverage_unique

# find minimal number of capsules that fit the image
@record_time("match", "hough") # time: ~ 1 sec / call
def match_shapes_to_image(im, edges_gray, circles, capsules, max_max_d = 3):
    print("    coloring shapes")
    
    # calculate number of nonzero points that both maps share in common
    def num_matching_pts(map_1, map_2):
        return ((map_1 * map_2) > 0).sum()

    def check_coverage_and_subtract(coverage_map):
        i, col_map = max(enumerate(color_maps),
                         key = lambda m: num_matching_pts(coverage_map, m[1]))
        if num_matching_pts(coverage_map, col_map) > \
           coverage_map.sum() * NEW_SHAPE_MIN_PERC_COVERAGE:
            col_map -= (coverage_map * col_map) > 0
            return i
        return None
    
    color_maps = decompose_colors(im, palette)
    coverage_hierarchy = [[calc_capsule_coverage(edges_gray, cap, d) * \
                           cap[2] * cap[3]
                           for d in range(max_max_d, -1, 1)]
                          for cap in capsules]

    matched_circles = []
    circle_colors = []
    for (cx, cy, cr) in circles:
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        cv2.circle(coverage_map, (int(cx), int(cy)), int(cr), 1, -1)
        i = check_coverage_and_subtract(coverage_map)
        if i != None:
            matched_circles.append((cx, cy, cr))
            circle_colors.append(i)
    
    matched_capsules = []
    capsule_colors = []
    for cap, cov in sorted(zip(capsules, coverage_hierarchy),
                           key = lambda cc: cc[1], reverse = True):
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        draw_capsule(coverage_map, *cap, 1, -1)
        i = check_coverage_and_subtract(coverage_map)
        if i != None:
            matched_capsules.append(cap)
            capsule_colors.append(i)
    return matched_circles, circle_colors, matched_capsules, capsule_colors

# construct color maps that a set of polygons constitute
@record_time("color_map")
def color_map_of_poly(circles, circle_colors, capsules, cap_colors):
    n = 4
    color_maps = [np.zeros((height, width), dtype = np.uint8) for _ in range(n)]

    # fill in color maps for circles, capsules
    for (cx, cy, cr), col in zip(circles, circle_colors):
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        cv2.circle(coverage_map, (int(cx), int(cy)), int(cr), 1, -1)
        color_maps[col] += coverage_map
    for (x, y, w, h, a), col in zip(capsules, cap_colors):
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        draw_capsule(coverage_map, x, y, w, h, a, 1, -1)
        color_maps[col] += coverage_map
    return color_maps

# remove polygons that are entirely contained within other polygons
@record_time("remove")
def remove_redundant(circles, circle_colors, capsules, cap_colors):
    print("   removing redundant polygons")
    
    color_maps = color_map_of_poly(circles, circle_colors, capsules, cap_colors)

    # determine if each circle contributes enough unique pixels
    circles_unique = []; circle_colors_unique = []
    for (cx, cy, cr), col in zip(circles, circle_colors):
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        cv2.circle(coverage_map, (int(cx), int(cy)), int(cr), 1, -1)
        if (color_maps[col] * coverage_map == 1).sum() > \
           coverage_map.sum() * SHAPE_MIN_PERC_UNIQUE:
            circles_unique.append((cx, cy, cr))
            circle_colors_unique.append(col)

    # determine if each capsule contributes enough unique pixels
    capsules_unique = []; capsule_colors_unique = []
    for (x, y, w, h, a), col in zip(capsules, cap_colors):
        coverage_map = np.zeros((height, width), dtype = np.uint8)
        draw_capsule(coverage_map, x, y, w, h, a, 1, -1)
        if (color_maps[col] * coverage_map == 1).sum() > \
           coverage_map.sum() * SHAPE_MIN_PERC_UNIQUE:
            capsules_unique.append((x, y, w, h, a))
            capsule_colors_unique.append(col)

    print("   circles removed: " + str(len(circles) - len(circles_unique)) + \
          ", capsules removed: " + str(len(capsules) - len(capsules_unique)))

    return circles_unique, circle_colors_unique, \
           capsules_unique, capsule_colors_unique

# refine parameter of circles, capsules, to cover edge map better
@record_time("refine")
def refine_poly(edges_gray, circles, capsules):
    print("   refining polygon parameters")
    circles_refined = []
    for (cx, cy, cr) in circles:
        # search local neighborhood of circle parameters
        opt_cir = None
        max_cov = None
        for dx, dy, dr in itertools.product(
            [-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2]):
            cir2 = (cx + dx, cy + dy, cr + dr)
            cov2 = [calc_circle_covered(edges_gray, *cir2, max_d = d)
                    for d in range(2, -1, -1)]
            cov2 = [sum(covered) / len(covered) for covered in cov2]
            if max_cov is None or cov2 > max_cov:
                opt_cir = cir2
                max_cov = cov2

        if MIN_CIRCLE_RADIUS <= opt_cir[2] <= MAX_CIRCLE_RADIUS:
            circles_refined.append(opt_cir)

    capsules_refined = []
    for (x, y, w, h, a) in capsules:
        # search local neighborhood of capsule parameters
        opt_cap = None
        max_cov = None
        for dx, dy, dw, dh, da in itertools.product(
            [-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2], [0]):
            cap2 = (x + dx, y + dy, w + dw, h + dh, a + da)
            cov2 = [calc_capsule_coverage(edges_gray, cap2, d)
                    for d in range(2, -1, -1)]
            if max_cov is None or cov2 > max_cov:
                opt_cap = cap2
                max_cov = cov2

        if opt_cap[2] > MIN_CAPSULE_WIDTH and \
           opt_cap[3] > MIN_CAPSULE_HEIGHT:
            capsules_refined.append(opt_cap)
    
    return circles_refined, capsules_refined

# calculate sum of iou (intersection over union) for each color channel
def color_map_iou(color_maps_1, color_maps_2):
    total_iou = 0
    e = 0.0000001
    for map_1, map_2 in zip(color_maps_1, color_maps_2):
        u = np.logical_and(map_1 > 0, map_2 > 0)
        i = np.logical_and(map_1 == map_2, u)
        total_iou += i.astype(np.uint64).sum() / (u.astype(np.uint64).sum() + e)
    return total_iou

# attempt to find polygons that are not covered by current polygons
@record_time("missing") # time ~ 20s
def hough_missing(im, circles, circle_colors, capsules, cap_colors):
    print("   finding missed polygons")
    color_maps = decompose_colors(im, palette)

    # remove pixels covered by the polygons that have already been found
    poly_maps = color_map_of_poly(circles, circle_colors, capsules, cap_colors)
    res_maps = [(color_maps[i] - poly_maps[i]) * (color_maps[i] > poly_maps[i])
                for i in range(len(color_maps))]

    # lenient hyperparameters
    MAX_CONSECUTIVE_MISSES = 1000
    CIRCLE_COVERAGE_TOL = 0.95
    SEMICIRCLE_COVERAGE_TOL = 0.80
    SEMICIRCLE_PARAM_DIFF_THRESHOLD = 3
    MIN_LINE_COVERAGE = 15
    CLOSEST_PT_COVERAGE_TOL = 0.75

    # find polygons in residual image
    im2 = recompose_image(res_maps)
    #write_hsv("r0_res.png", im2)
    edges2 = edge_map(im2)
    thin2 = thin_edges(edges2)
    thin_gray2 = cv2.cvtColor(cv2.cvtColor(thin2, cv2.COLOR_HSV2BGR),
                              cv2.COLOR_BGR2GRAY)
    circles2, circle_colors2, capsules2, cap_colors2 = \
              hough(im2, thin2, thin_gray2)

    # determine which new polygons improve iou
    # TODO: make sure new polygons are consistent with original edge map
    old_iou = color_map_iou(color_maps, poly_maps)
    num_circles_old = len(circles); num_caps_old = len(capsules)

    for circ, col in zip(circles2, circle_colors2):
        poly_maps2 = color_map_of_poly(circles + [circ], circle_colors + [col],
                                       capsules, cap_colors)
        new_iou = color_map_iou(color_maps, poly_maps2)
        if new_iou > old_iou:
            old_iou = new_iou
            circles = circles + [circ]
            circle_colors = circle_colors + [col]

    for cap, col in zip(capsules2, cap_colors2):
        poly_maps2 = color_map_of_poly(circles, circle_colors,
                                       capsules + [cap], cap_colors + [col])
        new_iou = color_map_iou(color_maps, poly_maps2)
        if new_iou > old_iou:
            old_iou = new_iou
            capsules = capsules + [cap]
            cap_colors = cap_colors + [col]

    print("   circles added: " + str(len(circles) - num_circles_old) + \
          ", capsules added: " + str(len(capsules) - num_caps_old))
    return circles, circle_colors, capsules, cap_colors

def read_tsv(fname, header = False):
    data = []
    with open(fname, mode = "r", encoding = "utf-8") as f:
        if header:
            head = f.readline()
        for line in f.readlines():
            line = line.strip("\r\n")
            if len(line) < 1:
                continue
            elif line.startswith("#"):
                continue
            data.append(line.split("\t"))
    return data

# remove all non monochrome colors in image (in-place)
def filter_monochrome(im):
    for y, x in np.ndindex(*im.shape[:2]):
        # hue should be 0 and saturation should be 0
        if im[y][x][0] > 0 or im[y][x][1] > 0:
            im[y][x] = np.array(black)
    return im

# parse image and determine the string that is being displayed
@record_time("read_font")
def read_font(im, font_to_template, pose_names = None):
    # obtain results of template matching for all given characters
    match_maps = []
    sorted_font_to_template = sorted(font_to_template.items())
    for font, (template, mask) in sorted_font_to_template:
        res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)
        res[np.isnan(res)] = 0
        match_maps.append(res)
    match_res = np.dstack(tuple(match_maps))

    # use repeated greedy matching until all pixels are identified
    char_locs = {}
    for _ in range(100):
        (y, x, i) = np.unravel_index(match_res.argmax(), match_res.shape)
        if match_res[y][x][i] > 0.1:
            char_locs[x] = sorted_font_to_template[i][0]
            match_res[y, x - 3 : x + 6, :] = 0
    return "".join([v for k, v in sorted(char_locs.items())])

# assign every pose to a series of frame in a video
@record_time("assign")
def assign_all():
    # get a list of all pose names
    pose_names = read_tsv("states.txt", header = True)
    pose_names = [row[1] for row in pose_names]

    # retrieve templates for all characters + digits
    font_to_template = {}
    for file in os.listdir("font"):
        font_char = file.partition(".")[0]
        template = cv2.cvtColor(cv2.imread("font/" + file), cv2.COLOR_BGR2HSV)
        template = cv2.bilateralFilter(template, 3, 10, 10)
        mask = np.zeros(template.shape, dtype = template.dtype)
        for y, x in np.ndindex(*template.shape[:2]):
            if template[y][x][2] > 8:
                mask[y][x] += 1
        font_to_template[font_char] = template, mask
    digit_to_template = {k : v for k, v in font_to_template.items()
                         if k.isdigit()}
    
    pose_to_frames = {}; strs = set()
    for file in os.listdir("videos/" + char):
        cap = cv2.VideoCapture("videos/" + char + "/" + file)
        num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pose_segment = frame[36 : 36 + 10, 210 : 210 + 160]
            pose_segment = cv2.bilateralFilter(pose_segment, 3, 10, 10)
            pose_segment = filter_monochrome(pose_segment)
            pose_string = read_font(pose_segment, font_to_template, pose_names)

            num_segment = frame[36 : 36 + 10, 372 : 372 + 60]
            num_segment = cv2.bilateralFilter(num_segment, 3, 10, 10)
            num_segment = filter_monochrome(num_segment)
            num_string = read_font(num_segment, digit_to_template, pose_names)

            if pose_string not in strs:
                strs.add(pose_string)
                print(pose_string)
                show_hsv(pose_segment)

                print(num_string)
                show_hsv(num_segment)

            num += 1
        cap.release()
    return pose_to_frames
# TODO: try tesseract
# preprocess image to get quantized image, edge map, grayscale edge map
@record_time("preprocess") # time: ~ half a second / call
def preprocess(frame):
    frame = cv2.bilateralFilter(frame, 5, 10, 10)
    frame = quantize_image(frame)
    edges = edge_map(frame)
    thin = thin_edges(edges)
    thin_gray = cv2.cvtColor(cv2.cvtColor(thin, cv2.COLOR_HSV2BGR),
                             cv2.COLOR_BGR2GRAY)
    return frame, thin, thin_gray

# use hough transform to find lines and semi/circles
@record_time("hough")
def hough(im, edges, edges_gray):
    print("   applying hough transforms")

    circles, circles_coverage, semicircles, semicircles_coverage = \
             find_circles(edges, edges_gray)

    lines = find_lines(edges_gray)
    
    capsules, capsule_cov = find_capsules(
        edges, edges_gray, circles, semicircles, lines)
    
    return match_shapes_to_image(im, edges_gray, circles, capsules)

def debug_poly(edges, circles, capsules, fname):
    edges_ = edges.copy()
    for (cx, cy, cr) in circles:
        out = cv2.circle(edges_, (int(cx), int(cy)), int(cr), blue, 1)
    for cap in capsules:
        out = draw_capsule(edges_, *cap, red, 1)
    write_hsv(fname, edges_)

def output_poly(circles, circle_colors, capsules, cap_colors, poly_file):
    with open(poly_file, mode = "w", encoding = "utf-8") as f:
        for (cx, cy, cr), col in zip(circles, circle_colors):
            f.write(str(col) + ", " + \
                    str((round(cx, 3), round(cy, 3), round(cr, 3))) + "\n")
        for cap, col in zip(capsules, cap_colors):
            f.write(str(col) + ", " + \
                    str(tuple(map(lambda x: round(x, 3), cap))) + "\n")

def hough_all():
    print("applying hough transforms for: " + char)
    for pose in os.listdir("frames/" + char)[16:]:
        if os.path.exists("poly/" + char + "/" + pose):
            continue
        
        print(" pose: " + str(pose))
        pose_dir = "frames/" + char + "/" + pose
        for file in os.listdir(pose_dir):
            print("  file: " + str(file))
            frame = cv2.cvtColor(cv2.imread(pose_dir + "/" + file),
                                 cv2.COLOR_BGR2HSV)
            frame = cv2.resize(frame, (584, 480), interpolation = cv2.INTER_CUBIC)
            frame, thin, thin_gray = preprocess(frame)
            circles, circle_colors, capsules, cap_colors = \
                     hough(frame, thin, thin_gray)

            circles, circle_colors, capsules, cap_colors = remove_redundant(
                circles, circle_colors, capsules, cap_colors)

            create_dir_if_not_exist("capsules/marth/" + pose)
            debug_poly(thin, circles, capsules,
                       "capsules/marth/" + pose + "/cap_" + file)

            create_dir_if_not_exist("poly/marth/" + pose)
            poly_file = "poly/marth/" + pose + "/poly_" +\
                        file.partition(".")[0] + ".txt"
            output_poly(circles, circle_colors, capsules, cap_colors, poly_file)

def refine_all():
    def debug(circles, circle_colors, capsules, cap_colors, title):
        poly_maps = color_map_of_poly(
            circles, circle_colors, capsules, cap_colors)
        im2 = recompose_image(poly_maps)
        write_hsv(title, im2)
    
    print("refining capsules for: " + char)
    for pose in os.listdir("poly/" + char):
        print(" pose: " + str(pose))
        pose_dir = "frames/" + char + "/" + pose
        poly_dir = "poly/" + char + "/" + pose
        for file in os.listdir(poly_dir):
            print("  file: " + str(file))
            circles, circle_colors, capsules, cap_colors = \
                     read_poly_file(poly_dir + "/" + file)
            frame_file = file.partition("_")[2].partition(".")[0] + ".png"
            frame = cv2.cvtColor(cv2.imread(pose_dir + "/" + frame_file),
                                 cv2.COLOR_BGR2HSV)
            frame, thin, thin_gray = preprocess(frame)
            #debug(circles, circle_colors, capsules, cap_colors, "r1_orig.png")

            # recolor the polygons
##            circles, circle_colors, capsules, cap_colors = match_shapes_to_image(
##                frame, thin_gray, circles, capsules, max_max_d = 1)
            #debug(circles, circle_colors, capsules, cap_colors, "r2_recol.png")

            # remove redundant polygons
##            circles, circle_colors, capsules, cap_colors = remove_redundant(
##                circles, circle_colors, capsules, cap_colors)
            #debug(circles, circle_colors, capsules, cap_colors, "r3_uniq.png")

            # refine parameters of each polygon
##            circles, capsules = refine_poly(thin_gray, circles, capsules)
##            debug(circles, circle_colors, capsules, cap_colors, "r4_refine.png")

            # try to find polygons that were previously missed
            circles, circle_colors, capsules, cap_colors = hough_missing(
                frame, circles, circle_colors, capsules, cap_colors)
##            debug(circles, circle_colors, capsules, cap_colors, "r5_missed.png")

            # remove redundant polygons again
            circles, circle_colors, capsules, cap_colors = remove_redundant(
                circles, circle_colors, capsules, cap_colors)
##            debug(circles, circle_colors, capsules, cap_colors, "r6_uniq2.png")

            debug_poly(thin, circles, capsules,
                       "capsules/marth/" + pose + "/cap_" + frame_file)

            poly_file = poly_dir + "/" + file
            output_poly(circles, circle_colors, capsules, cap_colors, poly_file)

if __name__ == "__main__":
    pose_to_frames = assign_all()
