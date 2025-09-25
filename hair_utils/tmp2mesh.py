import trimesh
import numpy as np
import json
from scipy.sparse import block_diag, diags
from scipy.sparse.linalg import spsolve

def compute_normals(points, lambda_=1.0):
    """
    Compute normals for a sequence of 3D points.
    :param points: Input sequence of 3D points, shape (N, 3), where N is the number of points
    :param lambda_: Smoothing term weight, controls the smoothness between normals
    :return: Sequence of normal vectors, shape (N, 3)
    """
    N = len(points)
    if N < 3:
        raise ValueError("At least 3 points are needed to compute normals")

    D_blocks = []
    for i in range(N):
        if i == 0:
            d_prev = np.zeros(3)
            d_next = points[i] - points[i + 1]
        elif i == N - 1:
            d_prev = points[i] - points[i - 1]
            d_next = np.zeros(3)
        else:
            d_prev = points[i] - points[i - 1]
            d_next = points[i] - points[i + 1]

        D_i = np.outer(d_prev, d_prev) + np.outer(d_next, d_next)
        D_blocks.append(D_i)

    data_blocks = [D_i + 2 * lambda_ * np.eye(3) for D_i in D_blocks]
    A_data = block_diag(data_blocks)

    smooth_blocks = [-lambda_, -lambda_]
    A_smooth = diags(smooth_blocks, offsets=[3, -3], shape=(3 * N, 3 * N))

    A = A_data + A_smooth

    cov = np.cov(points[:3].T)
    _, vecs = np.linalg.eigh(cov)
    n1 = vecs[:, 0]

    A_reduced = A[3:, 3:]
    b = -A[3:, :3] @ n1

    n_reduced = spsolve(A_reduced, b)

    normals = np.vstack([n1, n_reduced.reshape(-1, 3)])

    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    return normals


def create_mesh_from_template(vertices, width, thickness, shape, colorize=False):
    new_vertices = []
    new_faces = []
    new_vertex_colors = []

    normals = compute_normals(vertices)

    for i, vertex in enumerate(vertices):
        normal = normals[i]
        normal = normal / np.linalg.norm(normal)

        neighbor_idxs = [i-2, i-1, i, i+1, i+2]
        if neighbor_idxs[0] < 0:
            neighbor_idxs = [0, 1, 2, 3, 4]
        if neighbor_idxs[-1] >= len(vertices):
            neighbor_idxs = [len(vertices)-5, len(vertices)-4, len(vertices)-3, len(vertices)-2, len(vertices)-1]

        neighbors = vertices[neighbor_idxs]
        coeffs = np.array([-1/12, 2/3, 0, -2/3, 1/12])
        tangent = np.sum(coeffs[:, np.newaxis] * neighbors[-5:], axis=0)
        tangent /= np.linalg.norm(tangent)

        tangent -= np.dot(tangent, normal) * normal
        tangent = tangent / np.linalg.norm(tangent)

        tangent2 = np.cross(normal, tangent)

        # normal -> width, tangent2 -> thickness
        if shape == 'triangle':
            expanded_vertices = [
                vertex + width[i]/2 * normal,
                vertex + thickness[i]/3 * tangent2,
                vertex - width[i]/2 * normal,
                vertex - thickness[i]/6 * tangent2
            ]
            l = len(new_vertices)
            # [0,1,-4]; [1,2,-3]; [2,3,-2]; [3,0,-1]; [-3,-4,1]; [-4,-1,0]; [-1,-2,3]; [-2,-3,2]; [0,1,2]; [0,2,3]
            if l == 0:
                expanded_faces = [[0,2,1], [0,3,2]]
            else:
                expanded_faces = [
                    [l,l-4,l+1],
                    [l+1,l-3,l+2],
                    [l+2,l-2,l+3],
                    [l+3,l-1,l],
                    [l-3,l+1,l-4],
                    [l-4,l,l-1],
                    [l-1,l+3,l-2],
                    [l-2,l+2,l-3],
                    [l,l+1,l+2],
                    [l,l+2,l+3]
                ]

        elif shape == 'square':
            expanded_vertices = [
                vertex + width[i]/2 * normal,
                vertex + thickness[i]/2 * tangent2,
                vertex - width[i]/2 * normal,
                vertex - thickness[i]/2 * tangent2
            ]
            l = len(new_vertices)
            # [0,1,-4]; [1,2,-3]; [2,3,-2]; [3,0,-1]; [-3,-4,1]; [-4,-1,0]; [-1,-2,3]; [-2,-3,2]; [0,1,2]; [0,2,3]
            if l == 0:
                expanded_faces = [[0,2,1], [0,3,2]]
            else:
                expanded_faces = [
                    [l,l-4,l+1],
                    [l+1,l-3,l+2],
                    [l+2,l-2,l+3],
                    [l+3,l-1,l],
                    [l-3,l+1,l-4],
                    [l-4,l,l-1],
                    [l-1,l+3,l-2],
                    [l-2,l+2,l-3],
                    [l,l+1,l+2],
                    [l,l+2,l+3]
                ]

        new_vertices.extend(expanded_vertices)
        new_faces.extend(expanded_faces)

        if i == 0:
            new_vertex_colors.extend([[1, 0, 1]] * len(expanded_vertices))
        elif i == 1:
            new_vertex_colors.extend([[1, 0, 0]] * len(expanded_vertices))
        elif i == 2:
            new_vertex_colors.extend([[1, 1, 0]] * len(expanded_vertices))
        else:
            new_vertex_colors.extend([[0.5, 0.5, 0.5]] * len(expanded_vertices))

    if colorize:
        return np.array(new_vertices), np.array(new_faces), np.array(new_vertex_colors)
    return np.array(new_vertices), np.array(new_faces)


def hairtemplate2mesh(json_path, colorize=False):
    meshes = []

    with open(json_path) as f:
        data = json.load(f)

    for hair_tmp in data:
        shape = hair_tmp['shape']
        seq = hair_tmp['seq']

        vertices = np.array([s[:3] for s in seq])
        width = np.array([s[3] for s in seq])
        thickness = np.array([s[4] for s in seq])

        if len(vertices) < 3:
            print("Hair template has less than 3 vertices, skipping")
            continue

        if not colorize:
            new_vertices, new_faces = create_mesh_from_template(vertices, width, thickness, shape, colorize=colorize)
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
        else:
            new_vertices, new_faces, new_vertex_colors = create_mesh_from_template(vertices, width, thickness, shape, colorize=colorize)
            mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, vertex_colors=new_vertex_colors, process=False)
            
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        meshes.append(mesh)

    scene = trimesh.Scene(meshes)
    return scene
