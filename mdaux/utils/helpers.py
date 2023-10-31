import contextlib
import joblib
import numpy as np

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    check the function append_major_minor_axes_to_dpmdf for an example of how to use this"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def getSlopeIntercept(point_1, point_2):
    """Get the slope and intercept of a line given two points.
    
    Args:
        point_1 (list): first point
        point_2 (list): second point
        
    Returns:
        tuple: tuple of slope and intercept
    """
    m = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
    b = point_1[1] - m * point_1[0]
    return m, b

def getIntersection(m_1, b_1, m_2, b_2):
    """Get the intersection between two lines.
    
    Args:
        m_1 (float): slope of the first line
        b_1 (float): intercept of the first line
        m_2 (float): slope of the second line
        b_2 (float): intercept of the second line
        
    Returns:
        list: list of x and y coordinates of the intersection
    """
    x = (b_2 - b_1) / (m_1 - m_2)
    y = m_1 * x + b_1
    return np.array([x, y])

def set_numpy_random_seed(seed=42):
    np.random.seed(seed)

def line_orientation_angle(x, y):
    # get the point with the min y value
    i = np.argmin(y)

    # shift that point x and y to the origin
    x -= x[i]
    y -= y[i]

    return np.arctan2(y[1 - i], x[1 - i])

def compute_axes_of_polygon(vertices):
    """
    Computes the major and minor axes of a 2D polygon given its vertices.
    
    Parameters:
        vertices (numpy.ndarray): A N x 2 matrix representing the x and y coordinates of the vertices.
    
    Returns:
        major_axis_length (float): Length of the major axis.
        minor_axis_length (float): Length of the minor axis.
        major_axis_vector (numpy.ndarray): A vector spanning from centroid - major_axis to centroid + major_axis.
        minor_axis_vector (numpy.ndarray): A vector spanning from centroid - minor_axis to centroid + minor_axis.
    """
    
    # Step 1: Compute the centroid of the polygon
    centroid = np.mean(vertices, axis=0)
    
    # Step 2: Translate the vertices so the centroid is at the origin
    translated_vertices = vertices - centroid
    
    # Step 3: Compute the covariance matrix
    covariance_matrix = np.cov(translated_vertices, rowvar=False)
    
    # Step 4: Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Step 5: Sort eigenvalues and eigenvectors by eigenvalue in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 6: Compute major and minor axis lengths and vectors
    major_axis_length = np.sqrt(2 * sorted_eigenvalues[0])
    minor_axis_length = np.sqrt(2 * sorted_eigenvalues[1])
    
    major_axis_vector = np.vstack((centroid - major_axis_length * sorted_eigenvectors[:, 0],
                                   centroid + major_axis_length * sorted_eigenvectors[:, 0]))
    
    minor_axis_vector = np.vstack((centroid - minor_axis_length * sorted_eigenvectors[:, 1],
                                   centroid + minor_axis_length * sorted_eigenvectors[:, 1]))
    
    # Step 7: Compute the angles of the major and minor axes
    # TODO find a better way to do this because the current method alters the underlying axis vectors without copying (VERY BAD)
    major_axis_angle = line_orientation_angle(major_axis_vector[:, 0].copy(), major_axis_vector[:, 1].copy())
    minor_axis_angle = line_orientation_angle(minor_axis_vector[:, 0].copy(), minor_axis_vector[:, 1].copy())
    
    return major_axis_length, minor_axis_length, major_axis_vector, minor_axis_vector, major_axis_angle, minor_axis_angle