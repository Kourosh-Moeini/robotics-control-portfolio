import numpy as np
import torch
import numpy as np
import torch

def quat_integrate(q, w, dt):
    """
    Integrates a quaternion by an angular velocity vector.
    
    Args:
        q (np.ndarray): The initial orientation quaternion [w, x, y, z].
        w (np.ndarray): The angular velocity vector [wx, wy, wz].
        dt (float): The time step.

    Returns:
        np.ndarray: The new orientation quaternion.
    """
    w_pure = np.concatenate(([0], w))
    
    # Quaternion multiplication: q * w_pure
    w, x, y, z = q
    _, wx, wy, wz = w_pure
    
    dw = -x*wx - y*wy - z*wz
    dx =  w*wx + y*wz - z*wy
    dy =  w*wy - x*wz + z*wx
    dz =  w*wz + x*wy - y*wx
    
    q_dot = 0.5 * np.array([dw, dx, dy, dz])
    
    q_new = q + q_dot * dt

    if q_new[0]<0:
        q_new = -1*q_new
    
    # Re-normalize to maintain unit quaternion property
    return q_new / np.linalg.norm(q_new)

# def get_quat_error(q_target, q_current):
#     """
#     Calculates the scalar part 'w' of the error quaternion.
    
#     Args:
#         q_target (np.ndarray): The target orientation [w, x, y, z].
#         q_current (np.ndarray): The current orientation [w, x, y, z].

#     Returns:
#         float: The scalar 'w' part of the error quaternion.
#     """
#     # Conjugate of the current quaternion
#     q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    
#     # Error quaternion: q_error = q_target * conjugate(q_current)
#     w_target, x_target, y_target, z_target = q_target
#     w_conj, x_conj, y_conj, z_conj = q_current_conj
    
#     w_error = w_target*w_conj - x_target*x_conj - y_target*y_conj - z_target*z_conj
    
#     return w_error

def euler_zyx_to_quat(euler):
    """
    PyTorch version of Euler ZYX to Quaternion.
    
    Input: tensor of shape [..., 3]
           Column 2 -> Roll (X)
           Column 1 -> Pitch (Y)
           Column 1 -> Yaw (Z)
    
    Output: Quaternion tensor of shape [..., 4] as [w, qx, qy, qz]
    """
    z = euler[..., 0]  # Roll
    y = euler[..., 1]  # Pitch
    x = euler[..., 2]  # Yaw
    
    cz = torch.cos(z * 0.5)
    sz = torch.sin(z * 0.5)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cx = torch.cos(x * 0.5)
    sx = torch.sin(x * 0.5)
    
    w = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz    # ← fixed: cz, not cx
    
    return torch.stack([w, qx, qy, qz], dim=-1)

# def get_quat_error(q_target, q_current):
#     """
#     PyTorch version of Quaternion Error (Axis-Angle vector).
    
#     Returns: [Yaw_error, Pitch_error, Roll_error]
#              (Z, Y, X order based on validation results)
#     """
#     # Conjugate of current: [w, -x, -y, -z]
#     q_cur_conj = torch.cat([q_current[..., 0:1], -q_current[..., 1:]], dim=-1)
    
#     w1, x1, y1, z1 = q_target[..., 0], q_target[..., 1], q_target[..., 2], q_target[..., 3]
#     w2, x2, y2, z2 = q_cur_conj[..., 0], q_cur_conj[..., 1], q_cur_conj[..., 2], q_cur_conj[..., 3]
    
#     w = w1*w2 - x1*x2 - y1*y2 - z1*z2
#     x = w1*x2 + x1*w2 + y1*z2 - z1*y2
#     y = w1*y2 - x1*z2 + y1*w2 + z1*x2
#     z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
#     # Shortest path check (if w < 0, negate entire quaternion)
#     mask = w < 0
#     w = torch.where(mask, -w, w)
#     x = torch.where(mask, -x, x)
#     y = torch.where(mask, -y, y)
#     z = torch.where(mask, -z, z)
    
#     sin_half_theta = torch.sqrt(x*x + y*y + z*z)
    
#     # Calculate scale factor (2 * atan2(sin, cos) / sin)
#     # Using 'w' as cos_half_theta
#     val = torch.where(
#         sin_half_theta > 1e-6, 
#         (2.0 * torch.atan2(sin_half_theta, w)) / sin_half_theta, 
#         torch.tensor(2.0, device=w.device)
#     )
    
#     # Return ordered as [Z, Y, X] per requirement
#     return torch.stack([z*val, y*val, x*val], dim=-1)
