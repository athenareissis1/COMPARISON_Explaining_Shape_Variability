import numpy as np
import trimesh


class Mesh:
    """Minimal replacement for psbody.mesh.Mesh used in this project.

    Exposes:
      - mesh.v: (N, 3) float32 vertices
      - mesh.f: (F, 3) int32 faces (triangle indices)
      - mesh.compute_aabb_tree().nearest(points, return_parts)
    """

    def __init__(self, filename: str = None, v=None, f=None):
        # Case 1: construct from vertices/faces arrays (Mesh(v=..., f=...))
        if v is not None and f is not None:
            self.v = np.asarray(v, dtype=np.float32)
            self.f = np.asarray(f, dtype=np.int32)
            # internal trimesh for proximity queries
            self._tm = trimesh.Trimesh(
                vertices=self.v, faces=self.f, process=False
            )
            return

        # Case 2: construct from a file path (Mesh(filename="..."))
        if filename is None:
            raise ValueError("Mesh requires either filename=... or v and f arrays.")

        tm = trimesh.load(filename, process=False)
        if not isinstance(tm, trimesh.Trimesh):
            raise ValueError(f"Expected a triangular mesh in {filename}, got {type(tm)}")

        self._tm = tm
        self.v = np.asarray(tm.vertices, dtype=np.float32)
        self.f = np.asarray(tm.faces, dtype=np.int32)

    # ---- psbody-like API ----

    def compute_aabb_tree(self):
        """
        psbody.mesh.Mesh.compute_aabb_tree() -> object with .nearest(points, return_parts)

        Here we implement a simple nearest-face search in pure NumPy,
        to avoid depending on trimesh.proximity / rtree, and to have a
        stable return signature.
        """
        verts = self.v
        faces = self.f

        # Precompute triangle vertices (F, 3, 3) and centroids (F, 3)
        tri = verts[faces]           # (F, 3, 3)
        centroids = tri.mean(axis=1) # (F, 3)

        class _TreeWrapper:
            def __init__(self, tri, centroids):
                self._tri = tri
                self._centroids = centroids

            def nearest(self, points, return_parts):
                """
                points: (N, 3)
                returns: nearest_faces, nearest_parts, nearest_vertices
                nearest_parts is all zeros (we ignore edge/vertex classification).
                """
                points = np.asarray(points, dtype=np.float64)
                N = points.shape[0]

                # Compute squared distance from each point to each triangle centroid
                diff = self._centroids[None, :, :] - points[:, None, :]  # (N, F, 3)
                dist2 = (diff ** 2).sum(axis=2)                           # (N, F)
                nearest_face_idx = dist2.argmin(axis=1)                   # (N,)

                nearest_points = np.empty_like(points)
                for i in range(N):
                    f_id = nearest_face_idx[i]
                    tri = self._tri[f_id]  # (3, 3)
                    p = points[i]

                    # Project p onto triangle plane via least squares:
                    v0, v1, v2 = tri
                    e0 = v1 - v0
                    e1 = v2 - v0
                    A = np.stack([e0, e1], axis=1)  # (3, 2)
                    b = p - v0                      # (3,)
                    x, *_ = np.linalg.lstsq(A, b, rcond=-1)
                    a, b2 = x
                    proj = v0 + a * e0 + b2 * e1
                    nearest_points[i] = proj

                nearest_faces = nearest_face_idx.astype(np.int64)
                nearest_vertices = nearest_points.astype(np.float64)
                if return_parts:
                    nearest_parts = np.zeros(N, dtype=np.int64)
                else:
                    nearest_parts = None
                return nearest_faces, nearest_parts, nearest_vertices

        return _TreeWrapper(tri, centroids)