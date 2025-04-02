import imageio
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
#%% Define constants and parameters

n = 5 # for one single leg

# Frequency
omega_val = 2 * np.pi
omegaLF = np.ones(n) * omega_val
omegaRF = np.ones(n) * omega_val


# Convergence rate
mu_const = 1
mu = np.ones(n) * mu_const
# muLF = np.ones(n) * mu_const
# muRF = np.ones(n) * mu_const


# Amplitude
a_val = 10
a_paramLF = np.ones(n) * a_val
a_paramRF = np.ones(n) * a_val



# Tune the code after these:


# R_ampLF = np.array([30, 60, 45, 0 ,0.])
# R_ampRF = np.array([ -0, 60, -45, 0, -0.0])



# # For both simulation and hardware:  hop straight (to be tuned)
# R_ampLF = np.array([30, 60, 30, 0 ,0.])
# R_ampRF = np.array([-30, 60, -30, 0, -0.0])

# For both simulation and hardware:  go straight
R_ampLF = np.array([0, 60, 20, 0 ,0.])
R_ampRF = np.array([0, 60, -20, 0, -0.0])


# # For both simulation and hardware:  left turn
# R_ampLF = np.array([0, 60, 20, 0 ,0.])
# R_ampRF = np.array([ 0, 60, -20, 0, -0.0])*0.1


# # Mechanical offsets
# los = np.array([3, 3, 3, 3, 3])
# ros = np.array([-12, -12, -12, -12, -12])


# Joint offsets: For integration

# OffsetsLF = np.array([95, 0, 5, -70, -90])
# OffsetsRF = np.array([-100, 0, -5, -70, 85])


# # hop straight: hand tuning offsets(to be tuned)
# OffsetsLF = np.array([100, 0, -10, -75, -90])
# OffsetsRF = np.array([-100, 0, 5, -85, 90])




# Walking straight: hand tuning offsets(hardware)
OffsetsLF = np.array([50, -20, 0, -75, -90])
OffsetsRF = np.array([-50, -20, -5, -85, 90])

# # Walking straight: on hardware(hardware)
# OffsetsLF = np.array([90, 0, 0, -90, -90])*0.9
# OffsetsRF = np.array([-90, 0, -0, -90, 90])*1.1


# # Use these for simulation: fully symmetric
# OffsetsLF = np.array([90, 0, 0, -90, -90])
# OffsetsRF = np.array([-90, 0, -0, -90, 90])



# Desired phase differences

# high-level phase: LF, RF
# To change the initial condition of integration, as a high level CPG parameter
# phi_tilde_high = [0, 0]   # for two legs synchronized
phi_tilde_high = [0, np.pi]   # phase delays of between two legs: half of the period


# phi_tilde = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi,    0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi]  # forward： hop
phi_tilde_lf = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi]
phi_tilde_rf = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi]

theta_tilde_lf=[phi_tilde_lf[i+1]-phi_tilde_lf[i] for i in range(4)]
theta_tilde_rf=[phi_tilde_rf[i+1]-phi_tilde_rf[i] for i in range(4)]


# phi_tilde = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi,    np.pi, 2*np.pi, 2*np.pi, 1*np.pi, 1*np.pi]  # forward： walk

phi_tilde_lf = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi]
phi_tilde_rf = [0, 1*np.pi, 1*np.pi, 0*np.pi, 0*np.pi]
# phi_tilde_rf = [x + phi_tilde_high[1] for x in phi_tilde_rf]

theta_tilde_lf=[phi_tilde_lf[i+1]-phi_tilde_lf[i] for i in range(4)]
theta_tilde_rf=[phi_tilde_rf[i+1]-phi_tilde_rf[i] for i in range(4)]



# phi_tilde = [0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0]  # turning

# theta_tilde=[phi_tilde[i+1]-phi_tilde[i] for i in range(9)]

#%% ODE setup
phi0_lf = np.zeros(n)   # initial phases
r0_lf   = np.zeros(n)   # initial amplitudes
dr0_lf  = np.zeros(n)   # initial amplitude derivatives

x0_lf = np.concatenate((phi0_lf, r0_lf, dr0_lf))


phi0_rf = np.zeros(n) + phi_tilde_high[1]   # initial phases
r0_rf   = np.zeros(n)   # initial amplitudes
dr0_rf  = np.zeros(n)   # initial amplitude derivatives

x0_rf = np.concatenate((phi0_rf, r0_rf, dr0_rf))


# integration time step (change this if you want to change the speed)
dt  = 0.02

###########################################
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
t_span = (0, 200)  # simulate for 20 seconds
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
sol_lf = solve_ivp(
    lambda t, y: cpgODE(t, y, omegaLF, A, B, theta_tilde_lf, a_paramLF, R_ampLF, n),
    t_span,
    x0_lf,
    t_eval=t_eval
)

sol_rf = solve_ivp(
    lambda t, y: cpgODE(t, y, omegaRF, A, B, theta_tilde_rf, a_paramRF, R_ampRF, n),
    t_span,
    x0_rf,
    t_eval=t_eval
)


t_lf = sol_lf.t
x_lf = sol_lf.y.T  # shape (time_points, 3*n)

t_rf = sol_rf.t
x_rf = sol_rf.y.T  # shape (time_points, 3*n)



#%% Extract the results
phi_lf = x_lf[:, :n]          # phases
r_lf   = x_lf[:, n:2*n]       # amplitudes
dr_lf  = x_lf[:, 2*n:3*n]     # amplitude derivatives


#%% Extract the results
phi_rf = x_rf[:, :n]          # phases
r_rf   = x_rf[:, n:2*n]       # amplitudes
dr_rf  = x_rf[:, 2*n:3*n]     # amplitude derivatives


# Compute the rhythmic output signals: x_i = r_i * sin(phi_i)
desired_angle_lf = r_lf * np.sin(phi_lf) + np.ones(phi_lf.shape)*OffsetsLF
desired_angle_rf = r_rf * np.sin(phi_rf) + np.ones(phi_rf.shape)*OffsetsRF

normalize_min = -135
normalize_max = 135
###########################################

# Normalize the output from -1 to 1
desired_angle_lf_normalized = (desired_angle_lf - normalize_min) / (normalize_max - normalize_min) * 2 - 1
desired_angle_rf_normalized = (desired_angle_rf - normalize_min) / (normalize_max - normalize_min) * 2 - 1

# Change the decimal length
desired_angle_lf_normalized = np.round(desired_angle_lf_normalized,4) * np.array([1,6.4,1,1,1])
desired_angle_rf_normalized = np.round(desired_angle_rf_normalized,4) * np.array([1,-6.4,1,1,1])

print("desired_angle_lf_normalized ", desired_angle_lf_normalized, desired_angle_lf_normalized.shape )
print("desired_angle_rf_normalized ", desired_angle_rf_normalized, desired_angle_rf_normalized.shape )
###########################################
# Create a figure with n rows and 2 columns
fig, axs = plt.subplots(n, 2, figsize=(12, 2 * n), sharex='col', sharey=True)

start_idx = 0
end_idx=400

# Loop over each oscillator to plot its corresponding row
for i in range(n):
    # Left column: plots for the first dataset
    axs[i, 0].plot(t_lf[start_idx:end_idx], desired_angle_lf[start_idx:end_idx, i], linewidth=1.5)
    axs[i, 0].set_ylabel(f'x_{i+1}')
    axs[i, 0].grid(True)
    axs[i, 0].set_ylim(-135, 135)

    # Right column: plots for the second dataset
    axs[i, 1].plot(t_rf[start_idx:end_idx], desired_angle_rf[start_idx:end_idx, i], linewidth=1.5)
    axs[i, 1].set_ylabel(f'x_{i+1}')
    axs[i, 1].grid(True)
    axs[i, 1].set_ylim(-135, 135)

# Add a common x-label and a title for the entire figure
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
fig.suptitle('CPG Output Signals', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
###########################################

m = mujoco.MjModel.from_xml_path('6_forward_sucess_biped_snakebiped.xml')
d = mujoco.MjData(m)
m.vis.global_.offwidth = width = 720
m.vis.global_.offheight = height = 1008
renderer = mujoco.Renderer(m, width=width, height=height,max_geom= 50000)
render_length = 1400
renderer._scene.maxgeom = 50000
video_name = "6_forward_sucess_biped_snakebiped.mp4"
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

joint_values = []
actuation_values = []
time_stamps = []

# Main loop
for _ in range(render_length):  # Number of frames per simulation step

    # Control signals (apply normalized control signals to each joint)
    if idx % 3 == 0:

        d.ctrl[0] = desired_angle_lf_normalized[idx][0] * 1.0 # Apply normalized angles
        d.ctrl[1] = desired_angle_lf_normalized[idx][1] * 1.4
        d.ctrl[2] = desired_angle_lf_normalized[idx][2] * 1.0
        d.ctrl[3] = desired_angle_lf_normalized[idx][3] * 1.0 #1.94 for forward
        d.ctrl[4] = desired_angle_lf_normalized[idx][4] * 1.0 #1.94 for forward

        d.ctrl[5] = desired_angle_rf_normalized[idx][0] * 1.0 # Apply normalized angles
        d.ctrl[6] = desired_angle_rf_normalized[idx][1] * 1.4
        d.ctrl[7] = desired_angle_rf_normalized[idx][2] * 1.0
        d.ctrl[8] = desired_angle_rf_normalized[idx][3] * 1.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward
        d.ctrl[9] = desired_angle_rf_normalized[idx][4] * 1.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward

        # for k in range(0,len(d.ctrl)):
        #     if d.ctrl[k]>= 1.57 or d.ctrl[k]<=-1.57:
        #         print("joint ",i, "hit limit!")

    idx += 1
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
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(video_name+ '_MSE_.png', dpi=300, bbox_inches='tight')

# Close video writer
video_writer.close()
print("Video saved as ",video_name)

# idx = 0
# with mujoco.viewer.launch_passive(m, d) as viewer:
#      # Optionally, adjust the camera to see the entire simulation more clearly:
#      with viewer.lock():
#          viewer.cam.distance *= 10.0  # Increase the camera distance
#
#      # Start the simulation loop
#      sim_start_time = time.time()
#      while viewer.is_running():
#          # Calculate simulation elapsed time
#          current_sim_time = time.time() - sim_start_time
#
#          if idx % 3 == 0:
#          # d.ctrl[0: 5] = desired_angle_lf_normalized[idx]
#          # d.ctrl[5:10] = desired_angle_rf_normalized[idx]
#             d.ctrl[0] = desired_angle_lf_normalized[idx][0] * 1.0 # Apply normalized angles
#             d.ctrl[1] = desired_angle_lf_normalized[idx][1] * 1.4
#             d.ctrl[2] = desired_angle_lf_normalized[idx][2] * 1.0
#             d.ctrl[3] = desired_angle_lf_normalized[idx][3] * 1.0 #1.94 for forward
#             d.ctrl[4] = desired_angle_lf_normalized[idx][4] * 1.0 #1.94 for forward
#
#             d.ctrl[5] = desired_angle_rf_normalized[idx][0] * 1.0 # Apply normalized angles
#             d.ctrl[6] = desired_angle_rf_normalized[idx][1] * 1.4
#             d.ctrl[7] = desired_angle_rf_normalized[idx][2] * 1.0
#             d.ctrl[8] = desired_angle_rf_normalized[idx][3] * 1.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward
#             d.ctrl[9] = desired_angle_rf_normalized[idx][4] * 1.0 #1.94 for forward d.ctrl[9] = desired_angle_R1_normalized[idx][4] * 2.0 #1.94 for forward
#          idx+=1
#          if idx>=len(desired_angle_lf_normalized):
#              idx = 0
#
#          # Step the simulation forward
#          mujoco.mj_step(m, d)
#
#          # Update the viewer with the new simulation state
#          viewer.sync()
#
#          # Sleep for the remainder of the simulation timestep to match real time.
#          time.sleep(m.opt.timestep)
