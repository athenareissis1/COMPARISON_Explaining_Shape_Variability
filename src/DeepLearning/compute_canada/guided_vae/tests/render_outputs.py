import matplotlib
import torch
import torch.nn

from ..reconstruction.psbody_mesh_compat import Mesh

from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    BlendParams,
    HardGouraudShader
)

class Renderer:
    def __init__(self, img_size=256, rend_device=None):
        self.img_size = img_size
        self._rend_device = rend_device or torch.device('cuda')
        self.default_shader = HardGouraudShader(
            cameras=FoVPerspectiveCameras(),
            blend_params=BlendParams(background_color=[0, 0, 0])
        )
        self.simple_shader = ShadelessShader(
            blend_params=BlendParams(background_color=[0, 0, 0])
        )
        self._template_faces = None

    def set_renderings_size(self, size):
        self.img_size = size

    def set_rendering_background_color(self, color=None):
        color = [1, 1, 1] if color is None else color
        blend_params = BlendParams(background_color=color)
        self.default_shader.blend_params = blend_params
        self.simple_shader.blend_params = blend_params

    def errors_to_colors(self, values, min_value=None, max_value=None, cmap=None):
        device = values.device
        min_value = values.min() if min_value is None else min_value
        max_value = values.max() if max_value is None else max_value
        if min_value != max_value:
            values = (values - min_value) / (max_value - min_value)

        cmapper = matplotlib.cm.get_cmap(cmap)
        values = cmapper(values.cpu().detach().numpy(), bytes=True)
        return torch.tensor(values[:, :, :3]).to(device)

    def create_renderer(self):
        raster_settings = RasterizationSettings(image_size=self.img_size)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings,
                                      cameras=FoVPerspectiveCameras()),
            shader=self.default_shader
        )
        renderer.to(self._rend_device)
        return renderer

    def _get_template_faces(self):
        if self._template_faces is None:
            template_path = "/raid/compass/athena/data/PLY_friday_unified_meshes_subset_0_17/template/template.ply"
            template_mesh = Mesh(filename=template_path)
            self._template_faces = torch.as_tensor(template_mesh.f, dtype=torch.long)
        return self._template_faces.to(self._rend_device)

    def render(self, batched_data, vertex_errors=None, error_max_scale=None):
        renderer = self.create_renderer()
        batch_size = batched_data.shape[0]
        batched_verts = batched_data.detach().to(self._rend_device)

        if vertex_errors is not None:
            renderer.shader = self.simple_shader
            textures = TexturesVertex(
                self.errors_to_colors(
                    vertex_errors, min_value=0,
                    max_value=error_max_scale, cmap='plasma'
                ) / 255.0
            )
        else:
            renderer.shader = self.default_shader
            textures = TexturesVertex(torch.ones_like(batched_verts) * 0.5)

        template_faces = self._get_template_faces()  # [F, 3]
        meshes = Meshes(
            verts=batched_verts,
            faces=template_faces.unsqueeze(0).expand(batch_size, -1, -1),
            textures=textures,
        )

        cam_light_dist = 2.4
        rotation, translation = look_at_view_transform(
            dist=cam_light_dist, elev=0, azim=15)
        cameras = FoVPerspectiveCameras(R=rotation, T=translation,
                                        device=self._rend_device, znear=0.05)

        lights = PointLights(location=[[0.0, 0.0, cam_light_dist]],
                             diffuse_color=[[1., 1., 1.]],
                             device=self._rend_device)

        materials = Materials(shininess=0.5, device=self._rend_device)

        images = renderer(meshes, cameras=cameras, lights=lights,
                          materials=materials).permute(0, 3, 1, 2)
        return images[:, :3, ::]

    @staticmethod
    def compute_mse_loss(prediction, gt, reduction='mean'):
        return torch.nn.MSELoss(reduction=reduction)(prediction, gt)
    
    def compute_vertex_errors(self, out_verts, gt_verts):
        vertex_errors = self.compute_mse_loss(
            out_verts, gt_verts, reduction='none')
        vertex_errors = torch.sqrt(torch.sum(vertex_errors, dim=-1))
        vertex_errors *= 87
        return vertex_errors


class ShadelessShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = \
            blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        pixel_colors = meshes.sample_textures(fragments)
        images = hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images