import numpy as np


def create_spherical_ray(height, width):
    h = np.arange(height)
    theta = (height - h - 0.5) / height * np.pi
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    w = np.arange(width)
    phi = (width - w - 0.5) / width * 2 * np.pi + np.pi / 2
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    v = np.zeros((height, width, 3))
    ## interesting
    v[:, :, 0] = sin_theta.reshape((height, 1)) * cos_phi
    v[:, :, 1] = sin_theta.reshape((height, 1)) * sin_phi
    v[:, :, 2] = cos_theta.reshape((height, 1)) * np.ones(width)

    return v

def create_ptcloud(v, depthMapPlanes, depthMapIndices, numPlanes):
        height, width = 256, 512
        plane_indices = np.array(depthMapIndices)
        depth_map = np.zeros((height * width), dtype=np.float32)
        normal_map = np.zeros((height * width, 3), dtype=np.float32)
        v = v.reshape((height * width, 3))

        # index == 0 refers to the sky
        depth_map[np.nonzero(plane_indices == 0)] = np.nan
        # Create depth per plane
        for i in range(1, numPlanes):
            plane = depthMapPlanes[i]
            p_depth = np.ones((height * width)) * plane['d']

            vec = (plane['nx'], plane['ny'], plane['nz'])
            normal_map[np.nonzero(plane_indices == i), :] = vec

            depth = p_depth / v.dot(np.array((plane['nx'], plane['ny'], plane['nz'])))
            depth_map[np.nonzero(plane_indices == i)] = depth[np.nonzero(plane_indices == i)]
        depthMap = depth_map.reshape((height, width))
        return depthMap



v = create_spherical_ray(256, 512)
depthMap = create_ptcloud(create_ptcloud, depthMapPlanes, depthMapIndices, numPlanes)
