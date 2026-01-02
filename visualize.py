import open3d as o3d
import numpy as np
from graspnetAPI import GraspNet
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import voxel_sample_points, transform_points, create_table_points, collision_detection, compute_closest_points
from graspnetAPI.graspnet_eval import GraspNetEval

from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
import matplotlib.pyplot as plt
import trimesh
import os

# --- Global Config ---
graspnetroot = "/media/arjun/brein/graspnet"


# --- COPIED FUNCTIONS FROM YOUR SCRIPT (Mostly Unmodified) ---
# These are required to load the scene context and run the visualization

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

    ## collision detection
    if table is not None:
        scene = np.concatenate([scene, table])

    collision_mask_list, empty_list, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_dexgrasps=True)

    return grasp_list, dexgrasp_list, collision_mask_list



def get_dexgrasps(sceneId, annId, graspnetroot="/home/arjun/datasets/graspnet", camera="realsense"):
    """
    Loads all scene context.
    Returns obj_list (dexmodel_list), ggs_list (grasp_list), dgrasps_list (dexgrasp_list)
    """
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

    grasp_list, dexgrasp_list, collision_mask_list = conv_and_filter(
        grasp_group, model_sampled_list, dexmodel_list, pose_list, table=table_trans
    )
    
    # This matches the unpacking in your original main():
    # obj_list -> dexmodel_list
    # ggs -> grasp_list
    # dgrasps -> dexgrasp_list
    return dexmodel_list, grasp_list, dexgrasp_list


def find_contacts(grasp, obj, vis=False, ax=None, color='r'):

    if ax is None and vis:
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
        
    
    grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp.max_grasp_width_)
    num_samples = int(grasp.samples_per_grid * float(grasp_width_grid) / 2)
    
    g1_world, g2_world = grasp.endpoints

    
    if vis and ax is not None:
        approach_dist = 0.1
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
        ax.plot([end1[0], begin1[0]], [end1[1], begin1[1]], [end1[2], begin1[2]], color, linewidth=5)
        ax.plot([begin2[0], end2[0]], [begin2[1], end2[1]], [begin2[2], end2[2]], color, linewidth=5)
        ax.scatter(end1[0], end1[1], end1[2], s=80, c=color)
        ax.scatter(end2[0], end2[1], end2[2], s=80, c=color)
        
        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
        
        plt.draw()

    return c1, c2


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
    
    return wrist_center_meters, wrist_axis


# --- NEW VISUALIZATION SCRIPT ---

def visualize_from_npz(sceneId, annId):
    """
    Loads saved grasp pairs from an .npz file and visualizes them
    using the full intermediate pipeline (finding contacts, etc.).
    """
    
    # 1. Define paths
    npz_path = f"/home/arjun/dual-arm/results2/scene_{sceneId:03d}/ann_{annId:03d}_grasps.npz"
    
    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        print("Please make sure you have generated the dataset for this scene/annotation.")
        return

    # 2. Load the saved grasp pairs
    print(f"Loading saved grasps from: {npz_path}")
    try:
        grasp_group_array = np.load(npz_path, allow_pickle=True)['grasp_pairs']
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return
        
    num_pairs = len(grasp_group_array) // 2
    saved_grasp_pairs = grasp_group_array.reshape(num_pairs, 2, 17)
    print(f"Loaded {num_pairs} saved grasp pairs.")

    # 3. Re-load the scene context (Objects, DexNet grasps, GraspNet grasps)
    print("Loading scene context (this may take a moment)...")
    # obj_list = dexmodel_list, ggs_list = grasp_list, dgrasps_list = dexgrasp_list
    obj_list, ggs_list, dgrasps_list = get_dexgrasps(sceneId, annId, graspnetroot=graspnetroot)
    
    # 4. Initialize GraspNet API for point cloud loading
    g = GraspNet(graspnetroot, camera='realsense', split="test")
    
    # 5. Build a lookup map to match saved grasps to their DexNet objects
    print("Building grasp lookup map...")
    obj_grasp_maps = []
    for i in range(len(obj_list)):
        ggs = ggs_list[i]           # (M, 17) GraspNet arrays
        dgrasps = dgrasps_list[i]   # List of M DexNet objects or Nones
        
        trans_map = {}
        for k, dgrasp in enumerate(dgrasps):
            if dgrasp is not None:
                # Key: rounded translation (x,y,z)
                # Value: the DexNet grasp object
                gg_trans = ggs[k][12:15]
                trans_key = tuple(np.round(gg_trans, 5))
                trans_map[trans_key] = dgrasp
        
        obj_grasp_maps.append(trans_map)
    print("Map built.")

    # 6. Iterate and visualize
    if num_pairs == 0:
        print("No grasps to visualize.")
        return

    for i, pair in enumerate(saved_grasp_pairs):
        print(f"\n--- Visualizing Pair {i+1} of {num_pairs} ---")
        gg1, gg2 = pair[0], pair[1]
        
        # Get keys for lookup
        trans1_key = tuple(np.round(gg1[12:15].astype(float), 5))
        trans2_key = tuple(np.round(gg2[12:15].astype(float), 5))
        
        found_obj = None
        found_dexgrasp1 = None
        found_dexgrasp2 = None
        
        # Find which object this pair belongs to
        for obj_index, trans_map in enumerate(obj_grasp_maps):
            if trans1_key in trans_map and trans2_key in trans_map:
                found_obj = obj_list[obj_index]       # The GraspableObject3D
                found_dexgrasp1 = trans_map[trans1_key] # The ParallelJawPtGrasp3D
                found_dexgrasp2 = trans_map[trans2_key] # The ParallelJawPtGrasp3D
                print(f"Grasp pair matched to object index {obj_index}.")
                break
        
        if found_obj:
            # --- Visualization 1: Open3D (GraspNet-style) ---
            print("Showing Open3D visualization (GraspNet grasps)...")
            _6d_grasp = GraspGroup(np.array([gg1, gg2]))
            geometries = []
            geometries.append(g.loadScenePointCloud(sceneId=sceneId, annId=annId, camera='realsense'))
            geometries += _6d_grasp.to_open3d_geometry_list()
            o3d.visualization.draw_geometries(geometries)

            # --- Visualization 2: Matplotlib (DexNet-style contacts) ---
            # This is copied from your script's commented-out section
            print("Showing Matplotlib visualization (DexNet contacts)...")
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the contacts for each grasp
            c1a, c1b = find_contacts(found_dexgrasp1, found_obj, vis=True, ax=ax, color='g')
            c2a, c2b = find_contacts(found_dexgrasp2, found_obj, vis=True, ax=ax, color='r')

            # Plot the object surface points
            surface = found_obj.sdf.surface_points()[0]
            # Downsample for performance
            sample_size = min(surface.shape[0], 2000)
            surface = surface[np.random.choice(surface.shape[0], sample_size, replace=False)]
            
            ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], '.',
                       s=np.ones_like(surface[:, 0]) * 0.5, c='b', alpha=0.1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.title(f"DexNet Contacts (Pair {i+1})")
            
            # Try to set equal aspect ratio
            try:
                max_range = np.array([surface[:,0].max()-surface[:,0].min(), 
                                      surface[:,1].max()-surface[:,1].min(), 
                                      surface[:,2].max()-surface[:,2].min()]).max() / 2.0
                mean_x = surface[:,0].mean()
                mean_y = surface[:,1].mean()
                mean_z = surface[:,2].mean()
                ax.set_xlim(mean_x - max_range, mean_x + max_range)
                ax.set_ylim(mean_y - max_range, mean_y + max_range)
                ax.set_zlim(mean_z - max_range, mean_z + max_range)
            except Exception:
                # pass if surface is empty
                pass

            plt.show()
            plt.close(fig) # Close figure to free memory

        else:
            print(f"Warning: Could not find matching object for grasp pair {i+1}.")
            print(f"  Grasp 1 trans: {trans1_key}")
            print(f"  Grasp 2 trans: {trans2_key}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change these to visualize the scene/annotation you want
    SCENE_ID_TO_VISUALIZE = 1
    ANN_ID_TO_VISUALIZE = 200  # Change this to 0, 1, 2, ... 255
    
    visualize_from_npz(SCENE_ID_TO_VISUALIZE, ANN_ID_TO_VISUALIZE)
