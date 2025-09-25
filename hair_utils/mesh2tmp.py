import trimesh
import numpy as np
from sklearn.cluster import KMeans

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

def merge_close_vertices(mesh, tolerance=1e-8, dis_tol=1e-8):
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Find the unique vertices
    unique_vertices, inverse_indices = np.unique(
        np.round(vertices / tolerance).astype(int), 
        axis=0, 
        return_inverse=True
    )
    
    new_vertices = unique_vertices * tolerance
    new_faces = inverse_indices[faces]
    
    mesh = trimesh.Trimesh(
        vertices=new_vertices, 
        faces=new_faces
    )

    filtered = []
    filtered_idx = []
    for i in range(len(new_vertices)):
        if mesh.vertex_degree[i] != 6:
            filtered.append(new_vertices[i])
            filtered_idx.append(i)

    merge_sets = []
            
    for i in range(len(filtered)):
        for j in range(i+1, len(filtered)):
            if np.linalg.norm(filtered[i] - filtered[j]) < dis_tol:
                flag = False
                for k in merge_sets:
                    if i in k or j in k:
                        k.add(i)
                        k.add(j)
                        flag = True
                        break
                if not flag:
                    merge_sets.append({i, j})

    for merge_set in merge_sets:
        merge_set = list(merge_set)
        for i in range(1, len(merge_set)):
            new_faces[new_faces == filtered_idx[merge_set[i]]] = filtered_idx[merge_set[0]]
        new_vertices[filtered_idx[merge_set[0]]] = np.mean([new_vertices[filtered_idx[i]] for i in merge_set], axis=0)

    mesh = trimesh.Trimesh(
        vertices=new_vertices, 
        faces=new_faces
    )
    
    return mesh


def split_two_ends(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    filtered = []
    filtered_idx = []
    for i in range(len(vertices)):
        if mesh.vertex_degree[i] != 6:
            filtered.append(vertices[i])
            filtered_idx.append(i)

    group1_indices = [filtered_idx[0]]
    recheck = True
    while recheck:
        recheck = False
        for i in group1_indices:
            for j in filtered_idx:
                if j in group1_indices:
                    continue
                # check if j is neighbor of i, use faces
                for face in mesh.vertex_faces[i]:
                    if face == -1:
                        continue
                    if j in faces[face]:
                        group1_indices.append(j)
                        recheck = True
                        break

    group2_indices = [i for i in filtered_idx if i not in group1_indices]

    if len(group1_indices) == 1:
        print("special case", len(group1_indices), len(group2_indices), "retry kmeans")
        kmeans = KMeans(n_clusters=2, random_state=0).fit(filtered)
        group1_indices = [ filtered_idx[i] for i in np.where(kmeans.labels_ == 0)[0] ]
        group2_indices = [ filtered_idx[i] for i in np.where(kmeans.labels_ == 1)[0] ]

    if len(group1_indices) == 2 and len(group2_indices) == 4:
        # append 2 nearest vertices from 6-degree vertices to group1
        degree6_indices = [i for i in range(len(vertices)) if mesh.vertex_degree[i] == 6]
        dist_matrix = np.zeros((len(group1_indices), len(degree6_indices)))
        for i, idx1 in enumerate(group1_indices):
            for j, idx2 in enumerate(degree6_indices):
                dist_matrix[i, j] = np.linalg.norm(vertices[idx1] - vertices[idx2])
        # nearest 2!
        for i in range(2):
            if np.min(dist_matrix) > 1e-4:
                break
            idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            group1_indices.append(degree6_indices[idx[1]])
            dist_matrix = np.delete(dist_matrix, idx[1], axis=1)
    
    elif len(group2_indices) == 2 and len(group1_indices) == 4:
        # append 2 nearest vertices from 6-degree vertices to group2
        degree6_indices = [i for i in range(len(vertices)) if mesh.vertex_degree[i] == 6]
        dist_matrix = np.zeros((len(group2_indices), len(degree6_indices)))
        for i, idx1 in enumerate(group2_indices):
            for j, idx2 in enumerate(degree6_indices):
                dist_matrix[i, j] = np.linalg.norm(vertices[idx1] - vertices[idx2])
        # nearest 2!
        for i in range(2):
            if np.min(dist_matrix) > 1e-4:
                break
            idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            group2_indices.append(degree6_indices[idx[1]])
            print("special case", np.min(dist_matrix), idx)
            dist_matrix = np.delete(dist_matrix, idx[1], axis=1)

    elif len(group1_indices) == 9 and len(group2_indices) == 0:
        new_group1_indices = []
        degree = [ mesh.vertex_degree[i] for i in group1_indices ]
        bincount = np.bincount(degree)
        if bincount[4] == 3 and bincount[5] == 6:
            dist_matrix = np.zeros((len(group1_indices), len(group1_indices)))
            for i, idx1 in enumerate(group1_indices):
                for j, idx2 in enumerate(group1_indices):
                    dist_matrix[i, j] = np.linalg.norm(vertices[idx1] - vertices[idx2])
                    if i == j:
                        dist_matrix[i, j] = np.inf
            idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            new_group1_indices.append(group1_indices[idx[0]])
            dists = dist_matrix[idx[0]]
            for i in range(3):
                idx = np.argmin(dists)
                new_group1_indices.append(group1_indices[idx])
                dists[idx] = np.inf
            group1_indices = new_group1_indices
    
    return group1_indices, group2_indices


def check_end_type(mesh, grp):
    degree = [ mesh.vertex_degree[i] for i in grp ]
    bincount = np.bincount(degree)
    shape, top, subtops = None, None, None

    if bincount[3] == 4 and bincount.sum() == 4:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[3] == 3 and bincount.sum() == 3:
        shape = 'triangle'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[3] == 1 and bincount[4] == 3 and bincount.sum() == 4:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[3] == 1 and bincount[5] == 3 and bincount.sum() == 4:
        shape = 'triangle'
        top = grp[degree.index(3)]
        subtops = [ grp[i] for i in range(len(grp)) if degree[i] == 5 ]
    elif bincount[4] == 3 and bincount.sum() == 3:
        shape = 'triangle'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[4] == 1 and bincount[5] == 4 and bincount.sum() == 5:
        shape = 'square'
        top = grp[degree.index(4)]
        subtops = [ grp[i] for i in range(len(grp)) if degree[i] == 5 ]
    elif bincount[4] == 2 and bincount[5] == 2 and bincount.sum() == 4:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[3] == 2 and bincount[6] == 2 and bincount.sum() == 4:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[3] == 2 and bincount[6] == 1 and bincount.sum() == 3:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    elif bincount[5] == 2 and bincount[6] == 1 and bincount.sum() == 3:
        shape = 'square'
        top = None
        subtops = [ grp[i] for i in range(len(grp))]
    else:
        raise Exception('Unknown end type', bincount)

    return shape, top, subtops


def calc_width_thickness(points, normal, tol_ratio=0.1):
    if len(points) == 3:
        a, b, c = points
        lab, lbc, lca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
        tolerance = tol_ratio * np.mean([lab, lbc, lca])
        if np.abs(lab-lbc) < tolerance:
            width = lca
            proj = np.dot(b-a, c-a) / lca
            thickness = np.sqrt(lab**2 - proj**2)
        elif np.abs(lbc-lca) < tolerance:
            width = lab
            proj = np.dot(c-b, a-b) / lab
            thickness = np.sqrt(lbc**2 - proj**2)
        elif np.abs(lca-lab) < tolerance:
            width = lbc
            proj = np.dot(a-c, b-c) / lbc
            thickness = np.sqrt(lca**2 - proj**2)
        elif lab < 1e-6 and lbc < 1e-6 and lca < 1e-6:
            width = 0
            thickness = 0
        else:
            raise Exception('Not a isosceles triangle')

    elif len(points) == 4:
        a, b, c, d = points

        max_dis = 0
        max_st, max_ed = None, None
        for st, ed in [(a, b), (b, c), (c, d), (d, a), (c, a), (b, d)]:
            vec = ed - st
            dis = np.linalg.norm(vec)
            if dis > max_dis:
                max_dis = dis
                max_st = st
                max_ed = ed

        diagonal1_st, diagonal1_ed = max_st, max_ed
        diagonal2_st, diagonal2_ed = None, None
        for st, ed in [(a, b), (b, c), (c, d), (d, a), (c, a), (b, d)]:
            if np.all(st == max_st) or np.all(st == max_ed):
                continue
            if np.all(ed == max_st) or np.all(ed == max_ed):
                continue
            diagonal2_st, diagonal2_ed = st, ed
            break
        
        normal_min_cos = 1
        normal_max_cos = -1
        width_vec = None
        thickness_vec = None
        for vec in [diagonal1_ed-diagonal1_st, diagonal2_ed-diagonal2_st]:
            if np.abs(np.dot(vec, normal)) / np.linalg.norm(vec) > normal_max_cos:
                normal_max_cos = np.abs(np.dot(vec, normal)) / np.linalg.norm(vec)
                width_vec = vec
            if np.abs(np.dot(vec, normal)) / np.linalg.norm(vec) < normal_min_cos:
                normal_min_cos = np.abs(np.dot(vec, normal)) / np.linalg.norm(vec)
                thickness_vec = vec


        width_factor = normal_max_cos * (1-normal_min_cos**2) / ( normal_max_cos * (1-normal_min_cos**2) + normal_min_cos * (1-normal_max_cos**2) )
        thickness_factor = 1 - width_factor

        width = np.linalg.norm(width_vec) * width_factor + np.linalg.norm(thickness_vec) * thickness_factor
        thickness = np.linalg.norm(width_vec) * thickness_factor + np.linalg.norm(thickness_vec) * width_factor

    elif len(points) == 1:
        width = 0
        thickness = 0
    else:
        raise Exception('Unknown shape', len(points))

    return width, thickness


def regularize(mesh, grp1, grp2):
    vertices = mesh.vertices
    faces = mesh.faces

    visited = set()
    seq = []
    point_sets = []

    if np.var([vertices[i] for i in grp1]) < np.var([vertices[i] for i in grp2]):
        grp1, grp2 = grp2, grp1

    shape, top, subtops = check_end_type(mesh, grp1)

    if top is not None:
        seq.append([float(i) for i in vertices[top]])
        point_sets.append([top])
        visited.add(top)

    heaptop = set(subtops)

    while len(visited) < len(vertices):
        newheaptop = set()
        for i in heaptop:
            if i in visited:
                raise Exception('Visited vertex', i)
            visited.add(int(i))
        
        for i in heaptop:
            for j in mesh.vertex_neighbors[i]:
                if j in visited:
                    continue
                newheaptop.add(int(j))

        mean_pos = np.mean([vertices[i] for i in heaptop], axis=0)
        seq.append([float(i) for i in mean_pos])
        point_sets.append(list(heaptop))

        heaptop = newheaptop
        newheaptop = set()

    seq_array = np.array(seq)
    normals = compute_normals(seq_array)

    for i, point_set in enumerate(point_sets):
        width, thickness = calc_width_thickness([vertices[i] for i in point_set], normals[i])
        seq[i] += (width, thickness)

    return shape, seq


def mesh2hairtemplate(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump()
    else:
        meshes = [mesh]
    
    hairs = []
    for mesh in meshes:
        unique, group = trimesh.grouping.group_distance(mesh.vertices, 1e-8)
        index_map = np.zeros(len(mesh.vertices), dtype=int)
        for new_idx, group_indices in enumerate(group):
            index_map[group_indices] = new_idx
        merged_faces = index_map[mesh.faces]
        merged_mesh = trimesh.Trimesh(vertices=unique, faces=merged_faces)

        hair_ls = merged_mesh.split()
        hairs.extend(hair_ls)

    print('Total hairs:', len(hairs))

    # regularize each hair
    j = []
    for i, hair in enumerate(hairs):
        hair.update_faces(hair.unique_faces())
        # remove faces with 3 same vertices
        hair.update_faces(hair.nondegenerate_faces())
        grp1, grp2 = split_two_ends(hair)
        shape, seq = regularize(hair, grp1, grp2)
        d = {}
        d['shape'] = shape
        d['seq'] = seq
        j.append(d)

    return j, len(hairs)
