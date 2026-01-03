# Dual-Arm Grasp Dataset Generator

A pipeline for generating force-closure dual-arm grasp pairs from single-arm grasp databases. This system transforms parallel-jaw grasps from the GraspNet-1Billion dataset into coordinated dual-arm configurations that satisfy force closure constraints.

## Overview

This project implements a complete pipeline for dual-arm grasp generation and verification:

1. **Grasp Preprocessing**: Loads single-arm grasps from GraspNet, applies non-maximum suppression
2. **Object Assignment**: Assigns grasps to objects in cluttered scenes
3. **Collision Detection**: Filters collision-free grasps
4. **Pair Generation**: Creates candidate dual-arm grasp combinations
5. **Distance Pruning**: Eliminates kinematically infeasible configurations
6. **Force Closure Verification**: Validates grasp stability through convex optimization

## Installation

### Prerequisites

- Python 3.7+
- GraspNet-1Billion dataset
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install numpy
pip install scipy
pip install open3d
pip install cvxpy
pip install trimesh
pip install matplotlib
pip install tqdm
```

### GraspNetAPI Setup

The project includes a modified version of GraspNetAPI. To install:

```bash
cd graspnetAPI
pip install -e .
```

## Dataset Setup

1. Download the [GraspNet-1Billion dataset](https://graspnet.net/)
2. Update the `graspnetroot` path in the configuration files:
   - [generate.py](generate.py#L18)
   - [dataset.py](dataset.py#L19)
   - [visualize.py](visualize.py#L14)

```python
graspnetroot = "/path/to/your/graspnet"
```

## Usage

### Generate Dual-Arm Grasp Pairs

Generate dual-arm grasp pairs for a specific scene and annotation:

```python
from generate import generate_dual_arm_grasps

# Generate grasps for scene 1, annotation 0
scene_id = 1
ann_id = 0
grasp_pairs = generate_dual_arm_grasps(scene_id, ann_id)
```

To generate for multiple scenes/annotations:

```bash
python generate.py
```

### Visualize Grasps

Visualize generated dual-arm grasp pairs:

```python
from visualize import visualize_grasp_pair

visualize_grasp_pair(scene_id=1, ann_id=0, pair_index=0)
```

Or run the visualization script:

```bash
python visualize.py
```

### Dataset Creation

Create a complete dataset with force-closure verified grasp pairs:

```bash
python dataset.py
```

This will:
- Process all scenes and annotations
- Apply force closure optimization
- Save results to `results/` directory

## Project Structure

```
dual-arm/
├── README.md                          # This file
├── generate.py                        # Main grasp pair generation pipeline
├── dataset.py                         # Dataset creation and processing
├── visualize.py                       # Visualization tools
├── force_closure_optimization.py      # Force closure verification via cvxpy
├── for_mig.py                         # Migration utilities
├── graspnetAPI/                       # Modified GraspNetAPI library
│   ├── graspnetAPI/                   # Core API modules
│   ├── examples/                      # Usage examples
│   └── docs/                          # API documentation
├── docs/                              # Detailed documentation
│   └── doc.md                         # Complete theoretical analysis
└── results/                           # Generated grasp pairs
    ├── scene_001/
    │   └── ann_*_grasps.npz
    └── scene_009/
        └── ann_*_grasps.npz
```

## Core Components

### Force Closure Optimization

The force closure verification uses convex optimization to check if a dual-arm grasp can resist arbitrary external wrenches:

```python
from force_closure_optimization import fc_optimization

loss, forces = fc_optimization(
    contact_positions,   # 4x3 array of contact points
    contact_normals,     # 4x3 array of surface normals
    weight=1.0,          # Object weight
    friction_coeff=0.3,  # Friction coefficient
    soft_contact=False   # Contact model
)
```

### Grasp Representation

Each grasp is represented as a 17-dimensional vector:
- Score (1)
- Width, height, depth (3)
- Rotation matrix (9)
- Translation vector (3)
- Object ID (1)

### Distance-Based Pruning

Grasp pairs are pruned based on geometric constraints:
- Minimum gripper separation distance
- Maximum reach distance
- Collision avoidance

## Output Format

Generated grasp pairs are saved as `.npz` files containing:
- `grasp_pairs`: Array of dual-arm grasp configurations
- `force_closure_scores`: Force closure optimization results
- `contact_points`: Contact locations on the object
- `contact_normals`: Surface normals at contact points

## Documentation

For detailed theoretical background and mathematical formulation, see:
- [docs/doc.md](docs/doc.md) - Complete pipeline documentation with mathematical derivations

Topics covered:
- Grasp representation and parameterization
- Non-maximum suppression algorithms
- Collision detection methods
- Contact point computation
- Grasp map construction
- Force closure optimization theory

## Examples

### Basic Usage

```python
from generate import get_dexgrasps
from force_closure_optimization import fc_optimization

# Load scene data and grasps
scene_id = 1
ann_id = 0
dexmodels, grasp_lists, dexgrasp_lists = get_dexgrasps(scene_id, ann_id)

# Process each object
for obj_idx, dexgrasps in enumerate(dexgrasp_lists):
    print(f"Object {obj_idx}: {len(dexgrasps)} collision-free grasps")
```

### Force Closure Verification

```python
# Define contact points and normals
contacts = np.array([
    [0.1, 0.0, 0.0],   # Contact 1
    [-0.1, 0.0, 0.0],  # Contact 2
    [0.0, 0.1, 0.0],   # Contact 3
    [0.0, -0.1, 0.0]   # Contact 4
])

normals = np.array([
    [-1, 0, 0],  # Pointing inward
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0]
])

# Check force closure
loss, forces = fc_optimization(contacts, normals, weight=1.0)

if loss < 1e-5:
    print("Force closure achieved!")
else:
    print(f"Not force closure. Loss: {loss:.6f}")
```

## Related Datasets

- [GraspNet-1Billion](https://graspnet.net/) - Single-arm grasp dataset
- [DG16M](https://github.com/your-repo) - Dual-arm grasp dataset

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GraspNet-1Billion dataset and API
- Open3D for 3D visualization
- CVXPY for convex optimization
- Trimesh for mesh processing

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: graspnetAPI**
   - Install the included graspnetAPI: `cd graspnetAPI && pip install -e .`

2. **Dataset path errors**
   - Update `graspnetroot` variable in generate.py, dataset.py, and visualize.py

3. **CUDA out of memory**
   - Reduce batch size in processing
   - Process scenes individually

4. **Visualization not working**
   - Ensure Open3D is properly installed
   - Check display settings if running remotely

## Future Work

- [ ] Multi-threaded processing for faster generation
- [ ] Support for different gripper geometries
- [ ] Integration with motion planning
- [ ] Real robot validation
- [ ] Extended dataset with more scenes
