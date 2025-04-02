import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import imageio

#%% Define constants and parameters
n = 5  # number of oscillators

# Frequency
omega_val = 2 * np.pi
omega = np.ones(n) * omega_val

# Convergence rate
mu_const = 1
mu = np.ones(n) * mu_const

# Amplitude
a_val = 10
a_param = np.ones(n) * a_val

# Tune the code after these:

# Rolling
R_val = 75
# R_amp = np.ones(n) * R_val #Rotate
R_amp = np.array([1, -1, -1, 1, 1]) * R_val # Forward
# R_amp = np.array([3, -2, -1, 0.5, 0.25]) * R_val
# R_amp = np.array([3, -2, -1, 0.5, 0.25]) * R_val

# Desired phase differences
# theta_tilde = np.array([np.pi/2, -np.pi/2, np.pi/2, -np.pi/2]) #Rotate
theta_tilde = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]) #Forward
# Offset values of each motor
Offsets  = np.array([0, 0, 0, 0, 0])

#%% Matrices A and B
A = np.zeros((n, n))
A[0, 0] = -mu[0]
if n >= 2:
    A[0, 1] = mu[1]
for i in range(1, n - 1):
    A[i, i - 1] = mu[i]
    A[i, i]     = -2 * mu[i]
    A[i, i + 1] = mu[i]
A[n - 1, n - 2] = mu[n - 1]
A[n - 1, n - 1] = -mu[n - 1]

B = np.zeros((n, n - 1))
B[0, 0] = 1
for i in range(1, n - 1):
    B[i, i - 1] = -1
    B[i, i]     = 1
B[n - 1, n - 2] = -1

#%% ODE setup
phi0 = np.zeros(n)   # initial phases
r0   = np.zeros(n)   # initial amplitudes
dr0  = np.zeros(n)   # initial amplitude derivatives

x0 = np.concatenate((phi0, r0, dr0))

dt = 0.02  # time step
t_span = (0, 20)  # simulate for 20 seconds
t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]/dt)) # time points: based on dt and t_span
def cpgODE(t, x, omega, A, B, theta_tilde, a_param, R_amp, n):
    """
    Compute the derivatives for the CPG system.
    x is [phi, r, dr] with each block of length n.
    Returns: [dphi; dr; ddr]
    """
    phi = x[0:n]
    r   = x[n:2*n]
    dr  = x[2*n:3*n]

    # Phase derivatives
    dphi = omega + A.dot(phi) + B.dot(theta_tilde)

    # Second derivatives for amplitude
    ddr = a_param * ((a_param / 4) * (R_amp - r) - dr)

    # Note: The derivative of r is dr (not r)
    return np.concatenate((dphi, dr, ddr))
sol = solve_ivp(
    lambda t, y: cpgODE(t, y, omega, A, B, theta_tilde, a_param, R_amp, n),
    t_span,
    x0,
    t_eval=t_eval
)

t = sol.t
x = sol.y.T  # shape (time_points, 3*n)
phi = x[:, :n]          # phases
r   = x[:, n:2*n]       # amplitudes
dr  = x[:, 2*n:3*n]     # amplitude derivatives
desired_angle = r * np.sin(phi) + np.ones(phi.shape)*Offsets
normalize_min = -135
normalize_max = 135
desired_angle_normalized = (desired_angle - normalize_min) / (normalize_max - normalize_min) * 2 - 1
desired_angle_normalized = np.round(desired_angle_normalized,4)
# print("First 5 time points:", t[:5])
print("First 5 desired angle rows (each row corresponds to the outputs of all oscillators):")
print(desired_angle_normalized[:5, :], desired_angle_normalized.shape)

start_idx = 0
end_idx=3000
#%% Plot results using columns for each oscillator
plt.figure(figsize=(10, 8))
for i in range(n):
    plt.subplot(n, 1, i+1)
    # Here we plot the i-th column of output, which corresponds to oscillator i+1.
    # plt.plot(t, desired_angle_normalized[:, i], linewidth=1.5)
    plt.plot(t[start_idx:end_idx], desired_angle[start_idx:end_idx, i], linewidth=1.5)
    plt.ylabel(f'x_{i+1}')
    plt.grid(True)
    # plt.ylim(-1, 1)
    plt.ylim(-90, 90)
plt.xlabel('Time (s)')
plt.suptitle('CPG Output Signals', y=0.93)
plt.tight_layout()
plt.show()

############################# Renderer

m = mujoco.MjModel.from_xml_path('4_dual_snake_forward_ft.xml')
d = mujoco.MjData(m)
m.vis.global_.offwidth = width = 720
m.vis.global_.offheight = height = 1008
renderer = mujoco.Renderer(m, width=width, height=height,max_geom= 50000)
render_length = 1300
renderer._scene.maxgeom = 50000
video_name = "4_dual_snake_forward_ft.mp4"
video_writer = imageio.get_writer(video_name, fps=60, macro_block_size=16)

## NOTE: Name of cams
camera_name = "fixed_cam"  # Change to "fixed_cam" if needed
# camera_name = "tracking_cam"  # Change to "fixed_cam" if needed

cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
if cam_id == -1:
    print(f"Warning: Camera '{camera_name}' not found. Using default.")
else:
    print(f"Using predefined camera: {camera_name}")


if cam_id == -1:
    print(f"Warning: Camera '{camera_name}' not found in XML. Using default camera.")
else:
    print(f"Using predefined camera: {camera_name}")

# Find the COM body index
body_indices = np.arange(m.nbody)  # Indices of all bodies
body_mass = m.body_mass[body_indices]  # Mass of each body
total_mass = np.sum(body_mass)  # Total mass
def compute_com(data):
    """ Compute center of mass (COM) of the system """
    com = np.sum(data.xipos[body_indices] * body_mass[:, None], axis=0) / total_mass
    return com

# Get body IDs for two end-effectors
site_id_ee1 = m.body("m6_l").id  # First end-effector
site_id_ee2 = m.body("m6_r").id  # Second end-effector
print("location_to_track: \n", site_id_ee1)
print("location_to_track: \n", site_id_ee2)
# Storage for trajectories
traj_ee1 = []  # End-effector 1 trajectory
traj_ee2 = []  # End-effector 2 trajectory
traj_com = []  # Center of mass trajectory
max_traj_length = 5000  # Maximum number of stored positions

joint_values = []
actuation_values = []
time_stamps = []

############################# LOOP #############################
count=0
idx = start_idx
for i in range(m.njnt):
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(joint_name)

desired_angle_radiants = np.deg2rad(desired_angle)
# Main loop
for _ in range(render_length):  # Number of frames per simulation step

    # Control signals (apply normalized control signals to each joint)
    if count % 2 == 0:
        d.ctrl[0] = desired_angle_normalized[idx][0] * 1.4 # Apply normalized angles
        d.ctrl[1] = desired_angle_normalized[idx][1] * 1.0
        d.ctrl[2] = desired_angle_normalized[idx][2] * 1.0
        d.ctrl[3] = desired_angle_normalized[idx][1] * 2.0 #1.94 for forward
        d.ctrl[4] = desired_angle_normalized[idx][0] * 2.4 #1.94 for forward

        d.ctrl[5] = -d.ctrl[0]
        d.ctrl[6] = -d.ctrl[1]
        d.ctrl[7] = -d.ctrl[2]
        d.ctrl[8] = d.ctrl[3]
        d.ctrl[9] = d.ctrl[4]

        # d.ctrl[5] = desired_angle_normalized[idx][0] * 1.49 # Apply normalized angles
        # d.ctrl[6] = desired_angle_normalized[idx][1] * 1.49
        # d.ctrl[7] = desired_angle_normalized[idx][2] * 1.8
        # d.ctrl[8] = desired_angle_normalized[idx][1] * 1.94 #1.94 for forward
        # d.ctrl[9] = desired_angle_normalized[idx][0] * 1.94 #1.94 for forward
        for k in range(0,len(d.ctrl)):
            if d.ctrl[k]>= 1.57 or d.ctrl[k]<=-1.57:
                print("joint ",i, "hit limit!")
        # d.ctrl[3] = desired_angle_normalized[idx][3] * 1.57
        # d.ctrl[4] = desired_angle_normalized[idx][4] * 1.57
        idx += 1
    count += 1
    if idx >= end_idx:
        idx = start_idx
        print("reset\n")

    # Step the simulation forward
    mujoco.mj_step(m, d)
    if(d.time>7):
        # Get positions of both end-effectors and center of mass (COM)
        ee1_pos = d.xpos[site_id_ee1].copy()
        ee2_pos = d.xpos[site_id_ee2].copy()
        com_pos = d.subtree_com[0].copy()  # Index 0 gives total system COM
        # Append to trajectory lists
        traj_ee1.append(ee1_pos)
        traj_ee2.append(ee2_pos)
        traj_com.append(com_pos)
        joint_values.append(d.sensordata[:m.nsensor].copy())  # Store joint positions
        actuation_values.append(d.ctrl[:m.nu].copy())  # Store actuation values
        time_stamps.append(d.time)  # Store time

    # Keep only the last `max_traj_length` entries in the trajectory
    if len(traj_ee1) > max_traj_length:
        traj_ee1.pop(0)
        traj_ee2.pop(0)
        traj_com.pop(0)

    # First, update the scene from the simulation.
    renderer.update_scene(d, camera=camera_name)
    # Get the current number of simulation geoms (base index)
    base_index = renderer._scene.ngeom

    # Reshape data
    size = np.array([0.01, 0.01, 0.01], dtype=np.float64).reshape(3, 1)
    size_com = np.array([0.04, 0.04, 0.04], dtype=np.float64).reshape(3, 1)
    rgba_ee1 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)  # Red for EE1
    rgba_ee2 = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)  # Green for EE2
    rgba_com = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32).reshape(4, 1)  # Blue for COM
    mat = np.eye(3, dtype=np.float64).reshape(9, 1)

    # Remove oldest geoms if the buffer exceeds max_traj_length * 3
    while renderer._scene.ngeom > max_traj_length * 3:
        renderer._scene.ngeom -= 1

    # Draw trajectories for end-effectors and COM
    for i in range(1, len(traj_ee1)):
        if renderer._scene.ngeom < renderer._scene.maxgeom:
            geom = renderer._scene.geoms[renderer._scene.ngeom]

            # Draw EE1 trajectory (Red)
            pnt1 = np.array(traj_ee1[i - 1], dtype=np.float64).reshape(3, 1)
            pnt2 = np.array(traj_ee1[i], dtype=np.float64).reshape(3, 1)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, pnt1, mat, rgba_ee1)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, pnt1, pnt2)
            renderer._scene.ngeom += 1

            # Draw EE2 trajectory (Green)
            if i < len(traj_ee2):
                geom_ee2 = renderer._scene.geoms[renderer._scene.ngeom]
                pnt1_ee2 = np.array(traj_ee2[i - 1], dtype=np.float64).reshape(3, 1)
                pnt2_ee2 = np.array(traj_ee2[i], dtype=np.float64).reshape(3, 1)
                mujoco.mjv_initGeom(geom_ee2, mujoco.mjtGeom.mjGEOM_LINE, size, pnt1_ee2, mat, rgba_ee2)
                mujoco.mjv_connector(geom_ee2, mujoco.mjtGeom.mjGEOM_LINE, 2, pnt1_ee2, pnt2_ee2)
                renderer._scene.ngeom += 1

            # Draw COM trajectory (Blue)
            if i < len(traj_com):
                geom_com = renderer._scene.geoms[renderer._scene.ngeom]
                pnt1_com = np.array(traj_com[i - 1], dtype=np.float64).reshape(3, 1)
                pnt2_com = np.array(traj_com[i], dtype=np.float64).reshape(3, 1)
                mujoco.mjv_initGeom(geom_com, mujoco.mjtGeom.mjGEOM_LINE, size_com, pnt1_com, mat, rgba_com)
                mujoco.mjv_connector(geom_com, mujoco.mjtGeom.mjGEOM_LINE, 2, pnt1_com, pnt2_com)
                renderer._scene.ngeom += 1

    # # Update the renderer scene
    # renderer.update_scene(d)

    if(d.time>7):
        # Render the frame
        frame = renderer.render()

        # Write frame to video
        video_writer.append_data(np.flipud(frame))  # Flip vertically if needed

# Convert lists to NumPy arrays for proper slicing
joint_values = np.array(joint_values)
actuation_values = np.array(actuation_values)
time_stamps = np.array(time_stamps)
print(joint_values.shape)
print(actuation_values.shape)
print(time_stamps.shape)
# Compute Mean Squared Error (MSE)
mse_values = (actuation_values - joint_values) ** 2

mse_values =  np.sum(mse_values,axis=1)

time_stamps = time_stamps - time_stamps[0]

# for i in range(m.nu):
# plt.subplot(m.nu, 1, i+1)
# plt.plot(time_stamps, actuation_values[:, i], label=f'$q_{i+1}$ desired', linestyle='--', linewidth=1.5, color='r')
# plt.plot(time_stamps, joint_values[:, i], label=f'$q_{i+1}$ actual', linewidth=1.5, color='b')
plt.plot(time_stamps[:500], mse_values[:500], label='MSE', linewidth=1.5, color='b')
plt.ylabel(f'Joint {i+1}')
plt.legend()
plt.grid(True)

plt.xlabel('Time (s)')
plt.suptitle('Sensor vs Actuation Values')
plt.tight_layout()
plt.show()
plt.savefig(video_name+ '_MSE_.png', dpi=300, bbox_inches='tight')

# Close video writer
video_writer.close()
print("Video saved as ",video_name)
