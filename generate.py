import open3d as o3d
import numpy as np
from graspnetAPI import GraspNet
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import voxel_sample_points, transform_points, create_table_points, collision_detection, compute_closest_points
from graspnetAPI.graspnet_eval import GraspNetEval

from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
import matplotlib.pyplot as plt
from itertools import combinations
import concurrent.futures
from tqdm import tqdm
from force_closure_optimization import fc_optimization
import trimesh
import os
from functools import partial

# the format of numpy array is [score, width, height, depth, rotation_matrix(9), translation(3), object_id]

graspnetroot="/home/arjun/datasets/graspnet"

LOSS_THRESHOLD = 1e-5


def conv_to_trimesh(mesh3d):
    vertices = mesh3d.vertices
    faces = mesh3d.triangles
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return trimesh_mesh


def conv_and_filter(grasp_group, models, dexnet_models, poses, table=None):
    
    num_models = len(models)
    ## grasp nms
    grasp_group = grasp_group.nms(0.03, 30.0/180*np.pi)

    ## assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[indices]
    grasp_list = list()
    for i in range(num_models):
        grasp_i = grasp_group[model_to_grasp==i]
        grasp_list.append(grasp_i[:].grasp_group_array)

    # grasp_list = []
    # for i in range(num_models):
    #     grasp_list.append(pre_grasp_list[i])

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)

    return grasp_list, dexgrasp_list, collision_mask_list



def get_dexgrasps(sceneId, annId, graspnetroot="/home/arjun/datasets/graspnet", camera="realsense"):

    # print(f"Scene ID: {sceneId}, Annotation ID: {annId}")
    g = GraspNet(graspnetroot, camera, split="test")

    grasp_group = g.loadGrasp(sceneId, annId, format="6d", camera=camera).nms(0.03, 30.0/180*np.pi)

    eval = GraspNetEval(graspnetroot, camera=camera, split="test")

    model_list, dexmodel_list, obj_list = eval.get_scene_models(sceneId, annId)

    model_sampled_list = list()
    for model in model_list:
        model_sampled = voxel_sample_points(model, 0.008)
        model_sampled_list.append(model_sampled)

    _, pose_list, camera_pose, align_mat = eval.get_model_poses(sceneId, annId)
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

    grasp_list, dexgrasp_list, collision_mask_list = conv_and_filter(grasp_group, model_sampled_list, dexmodel_list, pose_list, table=table_trans)

    # print(np.array(dexgrasp_list, dtype=object).shape)
    # for i in range (len(dexgrasp_list)):
    #     print(f"Model {i}: Number of valid grasps: {len(dexgrasp_list[i])}")
    
    return dexmodel_list, grasp_list, dexgrasp_list

def find_contacts(grasp, obj, vis=False, ax=None, color='r', depth=None):

    if ax is None and vis:
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
        
    
    grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp.max_grasp_width_)
    num_samples = int(grasp.samples_per_grid * float(grasp_width_grid) / 2)
    
    g1_world, g2_world = grasp.endpoints

    
    if vis and ax is not None:
        approach_dist = depth if depth is not None else 0.1
        approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
        num_approach_samples = int(approach_dist_grid / 2)  # at least 1 sample per grid
        approach_axis = grasp.rotated_full_axis[:, 0]
        approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                    num_approach_samples, min_width=0)
        approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                    num_approach_samples, min_width=0)

        end1, end2 = approach_loa1[-1], approach_loa2[-1]
        begin1, begin2 = approach_loa1[0], approach_loa2[0]
        
        wrist = (end1 + end2) / 2.0
        
        # ax.scatter(grasp.center[0], grasp.center[1], grasp.center[2], s=80, c=color)
        ax.scatter(wrist[0], wrist[1], wrist[2], s=80, c='k')
        ax.plot([end1[0], end2[0]], [end1[1], end2[1]], [end1[2], end2[2]], color, linewidth=5)
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], color, linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], color, linewidth=5)


    line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, grasp.axis_, grasp.open_width, obj,
                                                                    num_samples, min_width=grasp.close_width)
    line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -grasp.axis_, grasp.open_width, obj,
                                                                    num_samples, min_width=grasp.close_width)

    c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=False)
    c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=False)
    
    begin1, begin2 = line_of_action1[0], line_of_action2[0]
    end1, end2 = obj.sdf.transform_pt_obj_to_grid(c1.point), obj.sdf.transform_pt_obj_to_grid(c2.point)

    if vis and ax is not None:
        # print(grasp.center)
        # print((end1+end2)/2)
        # print((c1.point + c2.point)/2)
        # print("=======")
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], color, linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], color, linewidth=5)
        ax.scatter(end1[0], end1[1], end1[2], s=80, c=color)
        ax.scatter(end2[0], end2[1], end2[2], s=80, c=color)
        
        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
        
        plt.draw()

    return c1, c2



def array_distance(g1_center, g2_center, g1_axis, g2_axis, alpha=0.05):
        center_dist = np.linalg.norm((g1_center - g2_center), axis=1)
        # axis = np.diagonal(np.abs(g1_axis.dot(g2_axis.T)))
        axis = np.sum((g1_axis * g2_axis), axis=1)
        dot_prod = np.maximum(np.minimum(axis, 1.0), 0.0)
        axis_dist = (2.0 / np.pi) * np.arccos(dot_prod)
        return center_dist + alpha * axis_dist
    

def find_wrist(grasp, obj):

    g1_world, g2_world = grasp.endpoints

    approach_dist = 0.1
    approach_dist_grid = obj.sdf.transform_pt_obj_to_grid(approach_dist)
    num_approach_samples = int(approach_dist_grid / 2)  # at least 1 sample per grid
    approach_axis = grasp.rotated_full_axis[:, 0]
    approach_loa1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, -approach_axis, approach_dist, obj,
                                                                num_approach_samples, min_width=0)
    approach_loa2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -approach_axis, approach_dist, obj,
                                                                num_approach_samples, min_width=0)

    end1, end2 = approach_loa1[-1], approach_loa2[-1]
    
    wrist_center_grid = (end1 + end2) / 2.0
    
    wrist_center_meters = obj.sdf.transform_pt_grid_to_obj(wrist_center_grid)
    
    wrist_axis = (end1 - end2) / np.linalg.norm(end1 - end2)
    
    # print(wrist_center_meters)
    
    return wrist_center_meters, wrist_axis


def pair_and_prune(dexgrasp_list, gg, c2c_threshold=0.05, w2w_threshold=0.1, obj=GraspableObject3D):
    
    dual_grasp = np.array([c for c in combinations(dexgrasp_list, 2)])
    dual_gg = np.array([c for c in combinations(gg, 2)])
    # np.random.shuffle(dual_grasp)
    print(f"Generated {len(dual_grasp)} dual grasps by pairing single-arm grasps.")
    
    grasp1_center = np.array([grasp1.center for i, (grasp1, grasp2) in enumerate(dual_grasp)])
    grasp2_center = np.array([grasp2.center for i, (grasp1, grasp2) in enumerate(dual_grasp)])
    grasp1_axis = np.array([grasp1.axis for i, (grasp1, grasp2) in enumerate(dual_grasp)])
    grasp2_axis = np.array([grasp2.axis for i, (grasp1, grasp2) in enumerate(dual_grasp)])

    grasp1_wrist = np.array([find_wrist(grasp1, obj)[0] for i, (grasp1, grasp2) in enumerate(dual_grasp)])
    grasp2_wrist = np.array([find_wrist(grasp2, obj)[0] for i, (grasp1, grasp2) in enumerate(dual_grasp)])


    c2c_dist = array_distance(grasp1_center, grasp2_center, grasp1_axis, grasp2_axis, alpha=0.00025)

    w2w_dist = np.array(np.linalg.norm(grasp1_wrist - grasp2_wrist, axis=1))

    indices_to_take = np.where((c2c_dist > c2c_threshold) & (w2w_dist > w2w_threshold))[0]
    # print(f"Skipping {len(dual_grasp) - len(indices_to_take)} dual grasps that are too close.")
    dual_grasp = dual_grasp[indices_to_take]
    dual_gg = dual_gg[indices_to_take]
    print(f'Found {len(dual_grasp)} dual-arm grasps.')

    return dual_grasp, dual_gg




def run_fc_optimization(mesh, contact_points, object_mass=6, friction_coeff=0.4, orientation = 0, num_workers=8):
    force_closure_passing_indices = []
    loss_values = []
    contact_forces = []
    frames = []
    
    contact_normals = np.zeros((len(contact_points), 4, 3))
    face1 = mesh.nearest.on_surface(contact_points[:, 0, :])[-1]
    face2 = mesh.nearest.on_surface(contact_points[:, 1, :])[-1]
    face3 = mesh.nearest.on_surface(contact_points[:, 2, :])[-1]
    face4 = mesh.nearest.on_surface(contact_points[:, 3, :])[-1]
    
    contact_normals[:, 0, :] = mesh.face_normals[face1]
    contact_normals[:, 1, :] = mesh.face_normals[face2]
    contact_normals[:, 2, :] = mesh.face_normals[face3]
    contact_normals[:, 3, :] = mesh.face_normals[face4]
    
    print("Contact normals found, starting optimization")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:   
        results = list(executor.map(fc_optimization, 
                                    contact_points, 
                                    contact_normals, 
                                    [10 * object_mass] * len(contact_points), 
                                    [friction_coeff] * len(contact_points), 
                                    [False] * len(contact_points),
                                    [orientation]*len(contact_points)))

    for i in tqdm(range(len(results))):
        f1, f2, f3, f4, loss, frame = results[i]
        if f1 is not None and f2 is not None and f3 is not None and f4 is not None:
            if loss < LOSS_THRESHOLD:
                force_closure_passing_indices.append(i)
            
        loss_values.append(loss)
        contact_forces.append([f1, f2, f3, f4])
        frames.append(frame)

    print(f'Total grasps passed: {len(force_closure_passing_indices)} out of {len(contact_points)}')
    
    loss_values = np.array(loss_values)
    contact_forces = np.array(contact_forces)
    frames = np.array(frames)
    
    return force_closure_passing_indices, loss_values, contact_forces, frames

def save_pairs(grasp_pairs, scores, sceneId, annId):

    for i in range(len(grasp_pairs)):
        grasp_pairs[i][0][0] = scores[i]
        grasp_pairs[i][1][0] = scores[i]
        
    grasp_pairs = grasp_pairs.flatten().reshape(-1,17)
    # print(grasp_pairs.shape)
    grasp_pairs = GraspGroup(grasp_pairs)
    
    save_path = f"/home/arjun/dual-arm/results2/scene_{sceneId:03d}/ann_{annId:03d}_grasps"
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(save_path, grasp_pairs=grasp_pairs.grasp_group_array)


#######################################################################################################################################
# --- Main Execution ---
#######################################################################################################################################

def main():
    """
    Main entry point for the dual-arm grasp processing script.
    """
    sceneId = 9

    for annId in range(256):
        print(f"\n--- Processing Scene {sceneId}, Annotation {annId} ---")
        obj_list, ggs, dgrasps = get_dexgrasps(sceneId, annId)
        grasp_pairs = np.array([]).reshape(0,2,17)
        scores = np.array([])

        print(f"Found {len(obj_list)} objects in scene.")
        for i in tqdm(range(len(obj_list)), desc=f"Object Loop (Ann {annId})"):

            gg = ggs[i]
            dg = dgrasps[i]
            obj = obj_list[i]

            filtered_pairs = [(d, gg) for d, gg in zip(dg, gg) if d is not None]
            dg = [pair[0] for pair in filtered_pairs]
            gg = [pair[1] for pair in filtered_pairs]


            dual_dexgrasps, dual_gg = pair_and_prune(dg, gg, 0.06, 0.1, obj)

            if len(dual_dexgrasps) == 0:
                dual_dexgrasps, dual_gg = pair_and_prune(dg, gg, 0.045, 0.1, obj)

            if len(dual_dexgrasps) == 0:
                # print(f"No valid dual grasps found for object {i}. Skipping.")
                continue

            contact_points = []
            for pair in dual_dexgrasps:
                grasp1, grasp2 = pair
                c1a, c1b = find_contacts(grasp1, obj)
                c2a, c2b = find_contacts(grasp2, obj)
                contact_points.append(np.array([c1a.point, c1b.point, c2a.point, c2b.point]))

            contact_points = np.array(contact_points)
            if contact_points.shape[0] == 0:
                # print(f"No contact points generated for object {i}. Skipping.")
                continue
            
            force_closure_passing_indices, loss_values, contact_forces, frames = run_fc_optimization(conv_to_trimesh(obj.mesh), contact_points, num_workers=15)

            if len(force_closure_passing_indices) > 2000:
                force_closure_passing_indices = np.random.choice(force_closure_passing_indices, 2000, replace=False)
            
            filtered = dual_gg[force_closure_passing_indices]
            grasp_pairs = np.concatenate((grasp_pairs, filtered))
            dexgrasp_pairs = dual_dexgrasps[force_closure_passing_indices]
            losses = loss_values[force_closure_passing_indices]
            scores = np.concatenate((scores, (LOSS_THRESHOLD - losses)*10000))
            
            for j in range(len(dexgrasp_pairs)):
                dual_dexgrasp = dexgrasp_pairs[j]
                fig = plt.gcf()
                ax = fig.add_subplot(111, projection='3d')
                grasp1, grasp2 = dual_dexgrasp[0], dual_dexgrasp[1]
                print(grasp_pairs[j][0], grasp_pairs[j][1])
                c1a, c1b = find_contacts(grasp1, obj, vis=True, ax=ax, color='g', depth=grasp_pairs[j][0][3])
                c2a, c2b = find_contacts(grasp2, obj, vis=True, ax=ax, color='r', depth=grasp_pairs[j][1][3])

                surface = obj.sdf.surface_points()[0]
                surface = surface[np.random.choice(surface.shape[0], 1000, replace=False)]
                scale = obj.sdf.transform_pt_grid_to_obj(surface[0]) / surface[0]
                ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                            s=np.ones_like(surface[:, 0]) * 0.3, c='b')

                plt.show()
                plt.clf()
            
        # if grasp_pairs.shape[0] > 0:
        #     print(f"Saving {grasp_pairs.shape[0]} grasp pairs for scene {sceneId}, ann {annId}")
        #     save_pairs(grasp_pairs, scores, sceneId, annId)
        # else:
        #     print(f"No passing grasps found for scene {sceneId}, ann {annId}. Nothing to save.")
            
    print("\n--- All Processing Complete ---")


# This is the standard entry point
if __name__ == "__main__":
    main()