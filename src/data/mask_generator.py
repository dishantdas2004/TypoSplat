"""
TypoSplat Letter Mask Generator
===============================
Projects the 3D typography mesh into 2D image space to create a binary mask.
Uses PyTorch3D's official OpenCV-to-PyTorch3D camera utility to prevent 
coordinate frame mismatches.
"""

import torch
import torch.nn.functional as F
from pytorch3d.io import load_ply
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

def get_letter_mask(ply_path, meta_json, device='cpu', target_size=(148, 148), orig_size=(518, 518)):
    """
    Loads a .ply mesh and metadata, projecting it to a binary mask.
    Outputs a downsampled [B, 1, H, W] tensor using adaptive max pooling 
    to preserve boundary pixels.
    """
    # 1. Load the Mesh
    verts, faces = load_ply(ply_path)
    verts = verts.to(device)
    faces = faces.to(device)
    mesh = Meshes(verts=[verts], faces=[faces])
    
    # 2. Extract Camera Matrices (World-to-Camera)
    # The metadata contains camera_to_world_matrix. We need the inverse (World-to-Camera)
    c2w = torch.tensor(meta_json["camera_to_world_matrix"], dtype=torch.float32, device=device)
    w2c = torch.linalg.inv(c2w)
    
    R_b = w2c[:3, :3]
    T_b = w2c[:3, 3]
    
    # 3. Blender to OpenCV Convention
    # Blender: +X right, +Y up, -Z forward
    # OpenCV: +X right, -Y up (+Y down), +Z forward
    # We flip Y and Z to bridge the gap before feeding to PyTorch3D's utility
    flip_yz = torch.tensor([[1., 0., 0.], 
                            [0., -1., 0.], 
                            [0., 0., -1.]], device=device)
    R_cv = flip_yz @ R_b
    T_cv = flip_yz @ T_b
    
    # PyTorch3D expects batched inputs
    R_cv = R_cv.unsqueeze(0) # [1, 3, 3]
    T_cv = T_cv.unsqueeze(0) # [1, 3]
    
    # 4. Construct Intrinsics Matrix (OpenCV format)
    fx, fy = meta_json["fx"], meta_json["fy"]
    cx, cy = meta_json["cx"], meta_json["cy"]
    camera_matrix = torch.tensor([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=torch.float32, device=device).unsqueeze(0)
    
    # 5. Use PyTorch3D's Official Utility to handle the PyTorch3D NDC conversion
    image_size = torch.tensor([orig_size], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(
        R=R_cv, 
        tvec=T_cv, 
        camera_matrix=camera_matrix, 
        image_size=image_size
    ).to(device)
    
    # 6. Rasterize the Mesh
    raster_settings = RasterizationSettings(
        image_size=orig_size[0], 
        blur_radius=0.0, 
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    fragments = rasterizer(mesh)
    
    # Pixels where pix_to_face >= 0 intersected with the mesh
    mask_518 = (fragments.pix_to_face[..., 0] >= 0).float() # [1, 518, 518]
    mask_518 = mask_518.unsqueeze(1) # [1, 1, 518, 518]
    
    # 7. Inclusive Downsampling
    # Adaptive max pool ensures we hit exactly 148x148 while dilating bounds
    mask_148 = F.adaptive_max_pool2d(mask_518, output_size=target_size)
    
    return mask_148