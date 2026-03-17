import torch
import os
import numpy as np
from glob import glob
# import openmesh as om
import trimesh

def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    # if vertices is not None:
    #     mesh = om.TriMesh(np.array(vertices), np.array(face))
    # else:
    #     n_vertices = face.max() + 1
    #     mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    # spirals = torch.tensor(
    #     extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    # return spirals

    faces = np.array(face, dtype=np.int64)
    if vertices is not None:
        verts = np.array(vertices, dtype=np.float32)
    else:
        n_vertices = face.max() + 1
        verts = np.ones([n_vertices, 3], dtype=np.float32)

    # trimesh mesh with vertices and triangular faces
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation)
    )
    return spirals
