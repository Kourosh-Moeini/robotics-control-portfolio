import numpy as np
from scipy.spatial.transform import Rotation as Rot
import warnings

class FrameConverter:
    def __init__(self, eul_t, T_t, tcp_offset):
        self.eul_t = eul_t
        self.T_t = T_t
        self.tcp_offset = tcp_offset
        self.R_t = self.eul2rotm(self.eul_t)

    def eul2rotm(self, eul, seq='ZYX'):
        r = Rot.from_euler(seq, eul)
        return r.as_matrix()

    def rotm2eul(self, rotm, seq='ZYX'):
        r = Rot.from_matrix(rotm)
        eul = r.as_euler(seq, degrees=False)
        
        # Handle gimbal lock dynamically
        if abs(eul[1] - np.pi/2) < 1e-6 and abs(eul[0] + np.pi/2) < 1e-6:
            eul[0] = 0.0
            eul[2] = eul[2] + np.pi/2 
        elif abs(eul[1] + np.pi/2) < 1e-6 and abs(eul[0] + np.pi/2) < 1e-6:
            eul[0] = 0.0
            eul[2] = eul[2] - np.pi/2
        
        return eul

    def local_global(self, p, eul=None):
        if eul is None:
            # eul = [np.deg2rad(-180), 0.0, np.deg2rad(60)]
            eul = [np.deg2rad(-180), 0.0, np.deg2rad(-60)]
            
        p = np.array(p).reshape(3, 1) 
        offset_vector = np.array([0, self.tcp_offset, 0]).reshape(3, 1)
        
        p_trans = self.R_t @ p + self.T_t + offset_vector
        p_trans = p_trans * 1000  # Convert meters to mm for the robot IK
        
        R_local = self.eul2rotm(eul, 'XYZ')
        R_trans = self.R_t @ R_local
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            eul_trans = self.rotm2eul(R_trans, 'ZYX')
        
        # Manipulator sequence: rx, ry, rz
        eul_manip = [eul_trans[2], eul_trans[1], eul_trans[0]]
        state_tcp = np.concatenate([p_trans.flatten(), eul_manip])
        
        return state_tcp.tolist()
