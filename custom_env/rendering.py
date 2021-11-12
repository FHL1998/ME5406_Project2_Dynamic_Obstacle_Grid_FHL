import math
import numpy as np


def down_sample(img, factor):
    """
    Down sample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img


def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def rotate_fn(fin, cx, cy, theta):
    def filter_out(x, y):
        x = x - cx
        y = y - cy

        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)

        return fin(x2, y2)

    return filter_out


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    direction_vector = p1 - p0
    dist = np.linalg.norm(direction_vector)
    unit_direction_vector = dir / dist  # unit vector

    x_min = min(x0, x1) - r
    x_max = max(x0, x1) + r
    y_min = min(y0, y1) - r
    y_max = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, unit_direction_vector)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(x_min, x_max, y_min, y_max):
    def fn(x, y):
        return x_min <= x <= x_max and y_min <= y <= y_max

    return fn


def point_in_triangle(a, b, c):
    """
    Compute barycentric coordinates (u, v) for point p with respect to triangle (a, b, c)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        denominator = (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) / denominator
        v = (dot00 * dot12 - dot01 * dot02) / denominator

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1
    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img
