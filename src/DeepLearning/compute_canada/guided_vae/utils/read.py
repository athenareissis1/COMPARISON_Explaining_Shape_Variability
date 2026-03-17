import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
# import openmesh as om
import trimesh

def read_mesh(path):
    # mesh = om.read_trimesh(path)
    # face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    # x = torch.tensor(mesh.points().astype('float32'))
    # edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    # edge_index = to_undirected(edge_index)
    
    # labels = torch.load("~/COMPARISON_Explaining_Shape_Variability/src/DeepLearning/compute_canada/guided_vae/data/CoMA/raw/torus/labels.pt")
    # subject = str(path.split("/")[-1].split(".")[0])
    # y = torch.Tensor([labels[subject]])


    # return Data(x=x, edge_index=edge_index, face=face, y=y)

    # load triangular mesh
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangular mesh in {path}, got {type(mesh)}")

    # faces: (3, F) long
    face = torch.from_numpy(mesh.faces.astype("int64")).T

    # vertices: (V, 3) float32
    x = torch.from_numpy(mesh.vertices.astype("float32"))

    # build edges from faces and make undirected
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)

    labels = torch.load(
        "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17/raw/labels.pt")
    subject = str(path.split("/")[-1].split(".")[0])
    y = torch.tensor([labels[subject]], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, face=face, y=y)
