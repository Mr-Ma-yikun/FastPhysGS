# TODO: Please refer to the original taichi implementation in PhysGaussian for the following code snippet. 
# The following code may be buggy or may not work as intended. 

import torch
import os
import numpy as np
import taichi as ti
import mcubes
from scipy.spatial import cKDTree

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def visualize_cluster(pcd, labels, cluster_id, output_file_path):
    # 提取当前聚类的点
    indices = np.where(labels == cluster_id)[0]
    cluster_pcd = pcd.select_by_index(indices)

    o3d.io.write_point_cloud(output_file_path, cluster_pcd, write_ascii=False)

def fill_convex_hull_with_points(hull_mesh, cluster_points, num_points=100000):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(hull_mesh)
    scene.add_triangles(mesh_t)

    # 获取包围盒用于采样
    aabb = hull_mesh.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # 在包围盒内随机采样点
    points = np.random.uniform(min_bound, max_bound, (num_points, 3))
    points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    # 查询哪些点在凸包内部
    occupancy = scene.compute_occupancy(points_tensor)
    inside_mask = occupancy.numpy() > 0.6  # occupancy: [0,1] 表示是否在内部
    inside_points = points[inside_mask]

    # -----------5. 计算每个内部候选点的重要性权重（基于原始点云密度）
    tree = cKDTree(cluster_points)
    distances, _ = tree.query(inside_points, k=1)  # 最近邻距离
    # 改gs参数做消融
    densities = np.exp(- (distances ** 2) / (2 * 0.02 ** 2))  # 高斯核密度权重，越大填的范围越多
    densities = np.maximum(densities, 1e-6)
    
    # 归一化为概率分布
    probabilities = densities / np.sum(densities)

    selected_indices = np.random.choice(
        np.arange(len(inside_points)),
        size=num_points,
        replace=True,  # 允许重复（防止候选不足）
        p=probabilities
    )
    selected_points = inside_points[selected_indices]

    # 转为点云
    filled_pcd = o3d.geometry.PointCloud()

    # 有重要性采样
    filled_pcd.points = o3d.utility.Vector3dVector(selected_points)
    return filled_pcd

def fill_HFPF(
        pos, obj_idxs, fili_nums
):
    # Step 1: Convert 3dgs points to Open3D
    device = pos.device
    pos_np = pos.detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_np)

    filtered_pcd = pcd

    # Step 2: Clustering
    print("Clustering...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(filtered_pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

    max_label = labels.max()
    for i in obj_idxs:
        visualize_cluster(filtered_pcd, labels, i, f"ori_{i}.ply")

    print(f"---------Found {max_label + 1} clusters-----------")

    # Step 3: Convex hull and filling

    print("Filling convex hulls...")
    filled_pcds = []
    filled_labels = []

    for cluster_id in obj_idxs:
        indices = np.where(labels == cluster_id)[0]
        cluster_points = np.asarray(filtered_pcd.points)[indices]

        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

        hull, _ = cluster_pcd.compute_convex_hull()
        if not hull.is_empty():
            filled_pcd = fill_convex_hull_with_points(hull, cluster_points, num_points=fili_nums[cluster_id])
            filled_pcds.append(filled_pcd)
            # 填充label
            filled_labels.append(np.full(len(filled_pcd.points), cluster_id)) 

    # Step4: Merge original and filled points and labels
    print("Merging points and labels...")
    new_points = pos_np
    for fill_pcd in filled_pcds:
        new_points = np.vstack((new_points, np.asarray(fill_pcd.points)))

    filled_labels = np.concatenate(filled_labels)
    all_labels = np.concatenate([labels, filled_labels])

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(new_points)
    
    for i in obj_idxs:
        visualize_cluster(merged_pcd, all_labels, i, f"filled_{i}.ply")

    points_array = np.asarray(merged_pcd.points)  # 形状: (N, 3)
    points_tensor = torch.from_numpy(points_array).to(device)
    points_tensor = points_tensor.to(dtype=torch.float32)

    return points_tensor, all_labels


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]


def init_filled_particles(pos, shs, cov, opacity, new_pos):
    # print('gaussian shape:', pos.shape)  # 查看输入形状
    # print('gaussian device:', pos.device)  # 是否为 cuda？
    # print('gaussian type:', pos.dtype)  # 是否为 float?

    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3).contiguous())
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)

    get_attr_from_closest(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_cov,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_cov,
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()

    opacity_tensor = torch.zeros_like(opacity_tensor).reshape(-1, 1)  # (M, 1)

    # 拼接原始点和新点的属性
    shs_tensor = torch.cat([shs, shs_tensor], dim=0)
    shs_tensor = shs_tensor.view(shs_tensor.shape[0], -1, 3)

    opacity_tensor = torch.cat([opacity, opacity_tensor], dim=0)  # 原始 + 新填充点
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    
    return shs_tensor, opacity_tensor, cov_tensor
