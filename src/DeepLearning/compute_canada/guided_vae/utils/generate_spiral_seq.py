# import openmesh as om
from sklearn.neighbors import KDTree
import numpy as np

def _build_vertex_adjacency(mesh):
    """Build adjacency list: for each vertex, list of neighbor vertex indices."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = int(faces.max()) + 1
    neighbors = [[] for _ in range(n_vertices)]

    for f in faces:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        # undirected edges
        for a, b in [(i, j), (j, k), (k, i)]:
            if b not in neighbors[a]:
                neighbors[a].append(b)
            if a not in neighbors[b]:
                neighbors[b].append(a)

    # keep a stable order
    neighbors = [sorted(nbrs) for nbrs in neighbors]
    return neighbors


def _next_ring(neighbors, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        after_last_ring = False
        # neighbors[vh1] plays role of mesh.vv(vh1)
        for vh2 in neighbors[vh1]:
            if after_last_ring:
                if is_new_vertex(vh2):
                    res.append(vh2)
            if vh2 in last_ring:
                after_last_ring = True
        for vh2 in neighbors[vh1]:
            if vh2 in last_ring:
                break
            if is_new_vertex(vh2):
                res.append(vh2)
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    """
    mesh: trimesh.Trimesh
    returns: list of vertex index sequences, length N_vertices,
             each sequence of length 'seq_length' after dilation.
    """
    neighbors = _build_vertex_adjacency(mesh)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    num_vertices = verts.shape[0]

    spirals = []
    for v_idx in range(num_vertices):
        reference_one_ring = list(neighbors[v_idx])
        spiral = [v_idx]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(neighbors, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(neighbors, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            # fallback: nearest neighbors in Euclidean space
            kdt = KDTree(verts, metric="euclidean")
            nn_idx = kdt.query(
                np.expand_dims(verts[spiral[0]], axis=0),
                k=seq_length * dilation,
                return_distance=False,
            ).tolist()
            spiral = [item for subspiral in nn_idx for item in subspiral]
        spirals.append(spiral[: seq_length * dilation][::dilation])
    return spirals

# ### OLD ###
# def _next_ring(mesh, last_ring, other):
#     res = []

#     def is_new_vertex(idx):
#         return (idx not in last_ring and idx not in other and idx not in res)

#     for vh1 in last_ring:
#         vh1 = om.VertexHandle(vh1)
#         after_last_ring = False
#         for vh2 in mesh.vv(vh1):
#             if after_last_ring:
#                 if is_new_vertex(vh2.idx()):
#                     res.append(vh2.idx())
#             if vh2.idx() in last_ring:
#                 after_last_ring = True
#         for vh2 in mesh.vv(vh1):
#             if vh2.idx() in last_ring:
#                 break
#             if is_new_vertex(vh2.idx()):
#                 res.append(vh2.idx())
#     return res


# def extract_spirals(mesh, seq_length, dilation=1):
#     # output: spirals.size() = [N, seq_length]
#     spirals = []
#     for vh0 in mesh.vertices():
#         reference_one_ring = []
#         for vh1 in mesh.vv(vh0):
#             reference_one_ring.append(vh1.idx())
#         spiral = [vh0.idx()]
#         one_ring = list(reference_one_ring)
#         last_ring = one_ring
#         next_ring = _next_ring(mesh, last_ring, spiral)
#         spiral.extend(last_ring)
#         while len(spiral) + len(next_ring) < seq_length * dilation:
#             if len(next_ring) == 0:
#                 break
#             last_ring = next_ring
#             next_ring = _next_ring(mesh, last_ring, spiral)
#             spiral.extend(last_ring)
#         if len(next_ring) > 0:
#             spiral.extend(next_ring)
#         else:
#             kdt = KDTree(mesh.points(), metric='euclidean')
#             spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
#                                               axis=0),
#                                k=seq_length * dilation,
#                                return_distance=False).tolist()
#             spiral = [item for subspiral in spiral for item in subspiral]
#         spirals.append(spiral[:seq_length * dilation][::dilation])
#     return spirals
