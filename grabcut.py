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

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    n_iter = 1  # remember to remove
    for i in range(n_iter):
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

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: we need to go over each pixel, give it a value from [0,k] with the label function, so that each pixel has an identifier in the format (alpha, [k]) and figure out how to recalculate GMM using these clusters
    from sklearn.mixture import GaussianMixture

    # # Assume you have already clustered your data into `num_clusters` clusters
    # # and saved the cluster assignments in a numpy array called `cluster_assignments`
    # # with shape (num_data_points,)
    # # You also have your original data saved in a numpy array called `data`
    #
    # for cluster_idx in range(num_clusters):
    #     # Extract the data points belonging to the current cluster
    #     cluster_data = data[cluster_assignments == cluster_idx]
    #
    #     # Fit a GaussianMixture model to the current cluster's data
    #     gm_model = GaussianMixture(n_components=num_components, covariance_type='full')
    #     gm_model.fit(cluster_data)

    bgGMM = GaussianMixture(n_components=5).fit(img[np.isin(mask, [GC_BGD, GC_PR_BGD])])
    fgGMM = GaussianMixture(n_components=5).fit(img[np.isin(mask, [GC_FGD, GC_PR_FGD])])

    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    edges, capacity, energy = calc_energy(img, mask, bgGMM, fgGMM)
    vertices = len(img[0]) * len(img) + 2
    g = igraph.Graph(vertices, edges)
    g.es["capacity"] = capacity

    cut = g.mincut(vertices - 2, vertices - 1, capacity=g.es["capacity"])
    min_cut = cut.partition

    return min_cut, energy


def update_mask(mincut_sets, mask):
    for fg in mincut_sets[0]:
        i, j = get_vertex_coordinates(fg, (len(mask), len(mask[0])))
        if mask[i][j] == GC_BGD or mask[i][j] == GC_FGD:
            continue
        else:
            mask[i][j] = GC_PR_FGD
    for bg in mincut_sets[1]:
        i, j = get_vertex_coordinates(fg, (len(mask), len(mask[0])))
        if mask[i][j] == GC_BGD or mask[i][j] == GC_FGD:
            continue
        else:
            mask[i][j] = GC_PR_BGD
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def calc_N_link_weight(pixel_1, pixel_2):
    dist = np.linalg.norm([pixel_1["row"], pixel_1["column"]] - [pixel_2["row"], pixel_2["column"]])
    beta = 0 #math.pow(np.mean(2 * np.square(<pixel_1["value"] - pixel_2["value"]>)), -1) #figure out what <> is
    return (50 / dist) * math.exp((-1) * beta * math.pow(np.linalg.norm(pixel_1["value"] - pixel_2["value"]), 2))

# link_type 0 means background, 1 means foreground
def calc_T_link_weight(pixel, pixel_type, values):
    bg_values = values["bg"]
    fg_values = values["fg"]
    n_components = len(bg_values["w"])

    if type == GC_BGD:
        return n_components, 0

    elif type == GC_FGD:
        return 0, n_components

    #calculate D_fore and D_back
    bg_sum = 0
    fg_sum = 0
    bg_values = values["bg"]
    fg_values = values["fg"]
    for i in range(len(bg_values["w"])):
        bg_f_1 = bg_values["w"][i] / math.sqrt(bg_values["det_c"][i])
        bg_f_2 = math.exp(np.matmul(np.matmul(0.5 * np.transpose(pixel - bg_values["m"][i]), bg_values["inv_c"][i]), (pixel - bg_values["m"][i])))
        bg_sum += bg_f_1 * bg_f_2

        fg_f_1 = fg_values["w"][i] / math.sqrt(fg_values["det_c"][i])
        fg_f_2 = math.exp(np.matmul(np.matmul(0.5 * np.transpose(pixel - fg_values["m"][i]), fg_values["inv_c"][i]), (pixel - fg_values["m"][i])))
        fg_sum += fg_f_1 * fg_f_2

    D_b = (-1) * math.log(bg_sum)
    D_f = (-1) * math.log(fg_sum)
    return D_f, D_b


def get_vertex_coordinates(vertex_id, dimensions):
    num_rows, num_cols = dimensions
    row = vertex_id // num_cols
    col = vertex_id % num_cols
    return row, col

def get_vertex_id(row, colunm, dimensions):
    return row * dimensions[0] + colunm
def get_neighbors(row, colunm, dimensions):
    neighbors = []
    for i in range(row - 1, row + 2):
        for j in range(colunm - 1, colunm + 2):
            if (0 <= i < dimensions[0] and 0 <= j < dimensions[i]) and (i != row or j != colunm):
                neighbors.append((i, j))
    return neighbors
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

    edges = []
    capacity = []
    SOURCE = len(img[0]) * len(img)
    SINK = SOURCE + 1
    V = 0
    U = 0

    for i in range(len(img)):
        for j in range(len(img[0])):
            pixel_vertex = get_vertex_id(i, j, (len(img), len(img[0])))
            source_weight, sink_weight = calc_T_link_weight(img[i, j], mask[i, j], values)
            U += source_weight + sink_weight  # build up U

            # add to edges and weights the respective lists in order
            edges.append((SOURCE, pixel_vertex))
            capacity.append(source_weight)

            edges.append((pixel_vertex, SINK))
            capacity.append(sink_weight)

            neighbors = get_neighbors(i, j, (len(img), len(img[0])))
            for n in neighbors:
                neighbor_vertex = get_vertex_id(n[0], n[1], (len(img), len(img[0])))
                weight = calc_N_link_weight(img[i, j], img[n[0], n[1]])
                edges.append((pixel_vertex, neighbor_vertex))
                capacity.append(weight)

                V += weight  # build up U

    return edges, capacity, U + V




    #U = np.sum([calc_T_link_weight(img[i][j], mask[i][j], 1, values) for i in range(len(img[0])) for j in range(len(img))])
    # print(U)


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
    neighbors = get_neighbors(1, 1, (len(img), len(img[0])))
    print(neighbors)
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
