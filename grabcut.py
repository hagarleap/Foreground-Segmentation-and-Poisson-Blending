import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph
import math

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)

    x, y = rect[:2]
    # The coordinates in the txt files are in absolute values
    w = rect[2] - x
    h = rect[3] - y

    # Initialize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    # for i in range(0, len(mask)):
    #     print(mask[i])

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def initalize_GMMs(img, mask, n_components=5):
    bgGMM = GaussianMixture(n_components=n_components).fit(img[np.isin(mask, [GC_BGD, GC_PR_BGD])])
    fgGMM = GaussianMixture(n_components=n_components).fit(img[np.isin(mask, [GC_FGD, GC_PR_FGD])])

    calc_energy(img, bgGMM, fgGMM)


    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    # flat_img = img.reshape(-1, 3)
    # predictions = np.round(bgGMM.predict_proba(flat_img), 3)
    # for row in predictions:
    #     print(row)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    g = igraph.Graph()
    # g.add_vertices(n_vertices)

    edges = []
    weights = []

    # g.add_edges(edges)
    # g.es['weight'] = weights

    min_cut = [[], []]
    energy = 0
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def calc_link_weight(pixel_1, pixel_2):
    # TODO: calc dist
    dist = 1
    beta = math.pow(np.mean(2 * np.square(pixel_1 - pixel_2)), -1)
    return (50 / dist) * math.exp((-1) * beta * math.pow(np.linalg.norm(pixel_1 - pixel_2), 2))


def calc_D(pixel, pixel_type, link_type, values):
    if type == GC_BGD:
        if link_type == 0:
            pass
        elif link_type == 1:
            pass
    elif type == GC_FGD:
        if link_type == 0:
            pass
        elif link_type == 1:
            pass
    elif type == GC_PR_BGD:
        if link_type == 0:
            pass
        elif link_type == 1:
            pass
    elif type == GC_PR_FGD:
        if link_type == 0:
            pass
        elif link_type == 1:
            pass

    # sum = 0
    # # for i in range(len(w)):
    # #     f_1 = w[i] / math.sqrt(det_c[i])
    # #     f_2 = math.exp(np.matmul(np.matmul(0.5 * np.transpose(pixel - m[i]), inv_c[i]), (pixel - m[i])))
    # #     sum += f_1 * f_2
    # D = (-1) * math.log(sum)
    # return D
    return 0

def calc_energy(img, mask, bgGMM, fgGMM):
    values = {
        "bg": {
            "w": bgGMM.weights_,
            "m": bgGMM.means_,
            "c": bgGMM.covariances_,
            "inv_c": np.linalg.inv(bgGMM.covariances_),
            "det_c": np.linalg.det(bgGMM.covariances_),
        },
        "fg": {
            "w": fgGMM.weights_,
            "m": fgGMM.means_,
            "c": fgGMM.covariances_,
            "inv_c": np.linalg.inv(fgGMM.covariances_),
            "det_c": np.linalg.det(fgGMM.covariances_),
        }
    }

    U = np.sum([calc_D(img[i][j], mask[i][j], 1, values) for i in range(len(img[0])) for j in range(len(img))])
    print(U)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


# Taken from https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
def draw_rect(path, x, y, w, h):
    image = cv2.imread(path)
    window_name = 'Image'
    start_point = (x, y)
    end_point = (x + w, y + h)
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)
    draw_rect(input_path, 16, 20, 620 - 16, 436 - 20)
    # Run the GrabCut algorithm on the image and bounding box
    # mask, bgGMM, fgGMM = grabcut(img, rect)
    # mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # # Print metrics only if requested (valid only for course files)
    # if args.eval:
    #     gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
    #     gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
    #     acc, jac = cal_metric(mask, gt_mask)
    #     print(f'Accuracy={acc}, Jaccard={jac}')
    #
    # # Apply the final mask to the input image and display the results
    # img_cut = img * (mask[:, :, np.newaxis])
    # cv2.imshow('Original Image', img)
    # cv2.imshow('GrabCut Mask', 255 * mask)
    # cv2.imshow('GrabCut Result', img_cut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
