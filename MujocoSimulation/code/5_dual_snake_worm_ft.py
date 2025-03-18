import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import imageio

#%% Define constants and parameters


#%% Define constants and parameters

n = 5 # for one single leg

# Frequency
omega_val = 2 * np.pi
omegaR1 = np.ones(n) * omega_val
omegaR2 = np.ones(n) * omega_val


# Convergence rate
mu_const = 1
muR1 = np.ones(n) * mu_const
muR1 = np.ones(n) * mu_const

mu = np.ones(n) * mu_const


# Amplitude
a_val = 10
a_paramR1 = np.ones(n) * a_val
a_paramR2 = np.ones(n) * a_val




# # snake Side-Winding

R_val = 25
# R_ampR1 = np.array([0.5, 0, 1.0, 0, -1.5])*R_val
# R_ampR2 = np.array([-1.5, 0, 1.0, 0, -3])*R_val

# # Snake forward (for both hardware and simulation)
# R_ampR1 = np.array([0, 1, 0, 1, 0])*R_val
# R_ampR2 = np.array([0, 1, 0, 1, 0])*R_val


# Snake forward 2
R_ampR1 = np.array([0, 30, 0, 30, 0])
R_ampR2 = np.array([0, 60, 0, -0, 0])



# # Snake turn
# R_ampR1 = np.array([1, -1, 1, -1, 1])*R_val
# R_ampR2 = np.array([1, -1, 1, -1, 1])*R_val



# # Mechanical offsets
# los = np.array([3, 3, 3, 3, 3])
# ros = np.array([-12, -12, -12, -12, -12])


# Joint offsets: For integration
# # snake forward (for hardware)
# OffsetsR1 = np.array([90, 0, -5, -0, -90])
# OffsetsR2 = np.array([90, 0, -5, -0, -90])


# # snake forward (for simulation)
# OffsetsR1 = np.array([90, 0, -0, -0, -90])
# OffsetsR2 = np.array([90, 0, -0, -0, -90])


# snake forward 2
# OffsetsR1 = np.array([0, -10, -0, 90, -0]) # on hardware
# OffsetsR2 = np.array([0, 60, 45, -45, -0]) # on hardware
OffsetsR1 = np.array([0, 0, 0, 0, 0])
OffsetsR2 = np.array([0, 0, 0, 0, 0])


# # snake turn
# OffsetsR1 = np.array([0, 0, -0, -0, -0])
# OffsetsR2 = np.array([0, 0, -0, -0, -0])

# Desired phase differences

# high-level phase: R1, R2
# To change the initial condition of integration, as a high level CPG parameter
phi_tilde_high = [0, 0]   # for two legs synchronized
# phi_tilde_high = [0, np.pi]   # phase delays of between two legs: haR1 of the period


# Low level CPG

# phi_tilde_R1 = [0, 0.5*np.pi, 1*np.pi, 0.5*np.pi, 0*np.pi]
# phi_tilde_R2 = [0, 0.5*np.pi, 1*np.pi, 0.5*np.pi, 0*np.pi]

phi_tilde_R1 = [0, 0.5*np.pi, 0, 0.5*np.pi, 0]
phi_tilde_R2 = [0, 0.5*np.pi, 0, 0, 0]


# # Desired phase differences
# theta_tilde = np.array([np.pi/2, 2*np.pi/2, np.pi/2, 0*np.pi/2])

theta_tilde_R1=[phi_tilde_R1[i+1]-phi_tilde_R1[i] for i in range(4)]
theta_tilde_R2=[phi_tilde_R2[i+1]-phi_tilde_R2[i] for i in range(4)]



# phi_tilde = [0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0]  # turning

# theta_tilde=[phi_tilde[i+1]-phi_tilde[i] for i in range(9)]

#%% ODE setup
phi0_R1 = np.zeros(n)   # initial phases
r0_R1   = np.zeros(n)   # initial amplitudes
dr0_R1  = np.zeros(n)   # initial amplitude derivatives

x0_R1 = np.concatenate((phi0_R1, r0_R1, dr0_R1))


phi0_R2 = np.zeros(n) + phi_tilde_high[1]   # initial phases
r0_R2   = np.zeros(n)   # initial amplitudes
dr0_R2  = np.zeros(n)   # initial amplitude derivatives

x0_R2 = np.concatenate((phi0_R2, r0_R2, dr0_R2))

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


dt = 0.05 # time step
t_span = (0, 20)  # simulate for 20 seconds
t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1]/dt)) # time points: based on dt and t_span

#%% ODE function definition
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

#%% Solve the ODE
sol_R1 = solve_ivp(
    lambda t, y: cpgODE(t, y, omegaR1, A, B, theta_tilde_R1, a_paramR1, R_ampR1, n),
    t_span,
    x0_R1,
    t_eval=t_eval
)

sol_R2 = solve_ivp(
    lambda t, y: cpgODE(t, y, omegaR2, A, B, theta_tilde_R2, a_paramR2, R_ampR2, n),
    t_span,
    x0_R2,
    t_eval=t_eval
)


t_R1 = sol_R1.t
x_R1 = sol_R1.y.T  # shape (time_points, 3*n)

t_R2 = sol_R2.t
x_R2 = sol_R2.y.T  # shape (time_points, 3*n)



#%% Extract the results
phi_R1 = x_R1[:, :n]          # phases
r_R1   = x_R1[:, n:2*n]       # amplitudes
dr_R1  = x_R1[:, 2*n:3*n]     # amplitude derivatives


#%% Extract the results
phi_R2 = x_R2[:, :n]          # phases
r_R2   = x_R2[:, n:2*n]       # amplitudes
dr_R2  = x_R2[:, 2*n:3*n]     # amplitude derivatives


# Compute the rhythmic output signals: x_i = r_i * sin(phi_i)
desired_angle_R1 = r_R1 * np.sin(phi_R1) + np.ones(phi_R1.shape)*OffsetsR1
desired_angle_R2 = r_R2 * np.sin(phi_R2) + np.ones(phi_R2.shape)*OffsetsR2

normalize_min = -135
normalize_max = 135
# Normalize the output from -1 to 1
desired_angle_R1_normalized = (desired_angle_R1 - normalize_min) / (normalize_max - normalize_min) * 2 - 1
desired_angle_R2_normalized = (desired_angle_R2 - normalize_min) / (normalize_max - normalize_min) * 2 - 1

# Change the decimal length
desired_angle_R1_normalized = np.round(desired_angle_R1_normalized,4)
desired_angle_R2_normalized = np.round(desired_angle_R2_normalized,4)

print("R1: ",desired_angle_R1_normalized, desired_angle_R1_normalized.shape)
print("R2: ",desired_angle_R2_normalized, desired_angle_R2_normalized.shape)

# Create a figure with n rows and 2 columns
fig, axs = plt.subplots(n, 2, figsize=(12, 2 * n), sharex='col', sharey=True)

start_idx = 0
end_idx=400
# Loop over each oscillator to plot its corresponding row
for i in range(n):
    # Left column: plots for the first dataset
    axs[i, 0].plot(t_R1[start_idx:end_idx], desired_angle_R1[start_idx:end_idx, i], linewidth=1.5)
    axs[i, 0].set_ylabel(f'x_{i+1}')
    axs[i, 0].grid(True)
    axs[i, 0].set_ylim(-135, 135)

    # Right column: plots for the second dataset
    axs[i, 1].plot(t_R2[start_idx:end_idx], desired_angle_R2[start_idx:end_idx, i], linewidth=1.5)
    axs[i, 1].set_ylabel(f'x_{i+1}')
    axs[i, 1].grid(True)
    axs[i, 1].set_ylim(-135, 135)

# Add a common x-label and a title for the entire figure
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
fig.suptitle('CPG Output Signals R1', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
############################# Renderer

m = mujoco.MjModel.from_xml_path('5_dual_snake_worm_ft.xml')
d = mujoco.MjData(m)
m.vis.global_.offwidth = width = 720
m.vis.global_.offheight = height = 1008
renderer = mujoco.Renderer(m, width=width, height=height,max_geom= 50000)
render_length = 4000
renderer._scene.maxgeom = 50000
video_name = "5_dual_snake_worm_ft.mp4"
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

############################# LOOP #############################
count=0
idx = start_idx
for i in range(m.njnt):
    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(joint_name)

joint_values = []
actuation_values = []
time_stamps = []

# Main loop
for _ in range(render_length):  # Number of frames per simulation step

    # Control signals (apply normalized control signals to each joint)
    if count % 2 == 0:

        d.ctrl[0] = desired_angle_R1_normalized[idx][4] * 3.0 # Apply normalized angles
        d.ctrl[1] = desired_angle_R1_normalized[idx][3] * 3.0
        d.ctrl[2] = desired_angle_R1_normalized[idx][2] * 3.0
        d.ctrl[3] = desired_angle_R1_normalized[idx][1] * 3.0 #1.94 for forward
        d.ctrl[4] = desired_angle_R1_normalized[idx][0] * 3.0 #1.94 for forward

        d.ctrl[5] = desired_angle_R2_normalized[idx][0] * 3.0 # Apply normalized angles
        d.ctrl[6] = desired_angle_R2_normalized[idx][1] * 3.0
        d.ctrl[7] = desired_angle_R2_normalized[idx][2] * 3.0
        d.ctrl[8] = desired_angle_R2_normalized[idx][3] * 3.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward
        d.ctrl[9] = desired_angle_R2_normalized[idx][4] * 3.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward

        # for k in range(0,len(d.ctrl)):
        #     if d.ctrl[k]>= 1.57 or d.ctrl[k]<=-1.57:
        #         print("joint ",i, "hit limit!")

        idx += 1
    count += 1
    if idx >= end_idx:
        idx = start_idx
        print("reset\n")

    # Step the simulation forward
    mujoco.mj_step(m, d)
    # Get positions of both end-effectors and center of mass (COM)
    ee1_pos = d.xpos[site_id_ee1].copy()
    ee2_pos = d.xpos[site_id_ee2].copy()
    com_pos = d.subtree_com[0].copy()  # Index 0 gives total system COM
    joint_values.append(d.sensordata[:m.nsensor].copy())  # Store joint positions
    actuation_values.append(d.ctrl[:m.nu].copy())  # Store actuation values
    time_stamps.append(d.time)  # Store time
    # Append to trajectory lists
    traj_ee1.append(ee1_pos)
    traj_ee2.append(ee2_pos)
    traj_com.append(com_pos)

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

    # Render the frame
    frame = renderer.render()

    # Write frame to video
    video_writer.append_data(np.flipud(frame))  # Flip vertically if needed

# Convert lists to NumPy arrays for proper slicing
joint_values = np.array(joint_values)
actuation_values = np.array(actuation_values)
time_stamps = np.array(time_stamps)
# Compute Mean Squared Error (MSE)
mse_values = (actuation_values - joint_values) ** 2

mse_values =  np.sum(mse_values,axis=1)
print(mse_values,mse_values.shape)
print(time_stamps,time_stamps.shape)

time_stamps = time_stamps - time_stamps[0]
plt.plot(time_stamps[:500], mse_values[:500], label='MSE', linewidth=1.5, color='b')
plt.ylabel(f'Joint {i+1}')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(video_name+ '_MSE_.png', dpi=300, bbox_inches='tight')

# Close video writer
video_writer.close()
print("Video saved as ",video_name)
