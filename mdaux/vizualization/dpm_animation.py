from mdaux.analysis.data_manipulation import (getAtStep, getDpmData, getVertexData, getParticleDataAsVector)
from mdaux.vizualization.general import (drawParticle, drawVector, RoundedPolygon, config_anim_plot, drawBoxBorders, 
                                         getValToColorMap, initialize_plot)
from mdaux.utils.helpers import getSlopeIntercept, getIntersection
from matplotlib import animation
import numpy as np
from tqdm import tqdm



def drawDpm(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, **kwargs):
    """Draw the dpm on the animation plot.
    
    Args:
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        vertex_df (dataframe): df containing the vertex data
        dpm_df (dataframe): df containing the dpm data
        dpm_id (int): index of the dpm
        step (int): step
        **kwargs: Additional keyword arguments for the add_patch function.
        
    Returns:
        None
    """
    vertex_df = getAtStep(getDpmData(vertex_df, dpm_id), step)
    dpm_df = getAtStep(getDpmData(dpm_df, dpm_id), step)
    com = vertex_df[['x', 'y']].mean().values
    com_mod = np.mod(com, box_lengths)
    com_diff = com_mod - com
    for vertex_id in vertex_df.id.unique():
        vertex = getVertexData(vertex_df, vertex_id, dpm_id)
        pos = getParticleDataAsVector(vertex, vertex_id, data_to_get='pos')
        pos = pos + com_diff
        radius = dpm_df.vertex_sigma.values[0] / 2
        drawParticle(ax_anim, pos, radius, **kwargs)

def drawDpmForces(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, **kwargs):
    """Draw the dpm forces on the animation plot.
    
    Args:
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        vertex_df (dataframe): df containing the vertex data
        dpm_df (dataframe): df containing the dpm data
        dpm_id (int): index of the dpm
        step (int): step
        **kwargs: Additional keyword arguments for the add_patch function.
        
    Returns:
        None
    """
    vertex_df = getAtStep(getDpmData(vertex_df, dpm_id), step)
    dpm_df = getAtStep(getDpmData(dpm_df, dpm_id), step)
    com = vertex_df[['x', 'y']].mean().values
    com_mod = np.mod(com, box_lengths)
    com_diff = com_mod - com
    for vertex_id in vertex_df.id.unique():
        vertex = getVertexData(vertex_df, vertex_id, dpm_id)
        pos = getParticleDataAsVector(vertex, vertex_id, data_to_get='pos')
        pos = pos + com_diff
        force = getParticleDataAsVector(vertex, vertex_id, data_to_get='force')
        drawVector(ax_anim, pos, force, **kwargs)


def drawDpmAsPolygon(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, draw_images, **kwargs):
    """Draw the dpm as a polygon on the animation plot.
    
    Args:
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        vertex_df (dataframe): df containing the vertex data
        dpm_df (dataframe): df containing the dpm data
        dpm_id (int): index of the dpm
        step (int): step
        box_lengths (list): list of box lengths
        use_image (bool): include the nearest image of the dpm (default: False)
        **kwargs: Additional keyword arguments for the add_patch function.
        
    Returns:
        None
    """
    # get the relevant data
    vertex_df = getAtStep(getDpmData(vertex_df, dpm_id), step)
    dpm_df = getAtStep(getDpmData(dpm_df, dpm_id), step)

    radius = dpm_df.vertex_sigma.values[0] / 2

    # get the center of mass of the vertices
    com = vertex_df[['x', 'y']].mean().values
    com_mod = np.mod(com, box_lengths)
    com_diff = com_mod - com

    # construct the segment list
    segment_list = []
    for vertex_id in vertex_df.id.unique():
        vertex = getVertexData(vertex_df, vertex_id, dpm_id)
        pos = getParticleDataAsVector(vertex, vertex_id, data_to_get='pos')

        vertex_id_next = vertex_id + 1 if vertex_id < vertex_df.id.max() else 0
        vertex_next = getVertexData(vertex_df, vertex_id_next, dpm_id)
        pos_next = getParticleDataAsVector(vertex_next, vertex_id_next, data_to_get='pos')

        # get the unit vector perpindicular to the segment
        segment = pos_next - pos
        segment = segment / np.linalg.norm(segment)
        offset = - np.array([-segment[1], segment[0]])

        # offset pos and pos_next by radius in the direction of the offset
        pos = pos + offset * radius
        pos_next = pos_next + offset * radius

        # put the segment into the segment list
        segment_list.append([pos, pos_next])

    # find the intersection between neighbouring segments
    intersections = []
    for i in range(len(segment_list)):
        k = i + 1 if i < len(segment_list) - 1 else 0  # get the next segment

        # get the intersection between the lines defined by the coordinates in the segments
        m_1, b_1 = getSlopeIntercept(segment_list[i][0], segment_list[i][1])
        m_2, b_2 = getSlopeIntercept(segment_list[k][0], segment_list[k][1])

        # Check for vertical lines and parallel lines
        if m_1 is None and m_2 is None:
            # Both lines are vertical
            intersection = None
        elif m_1 is None:
            # First line is vertical, find intersection
            x = segment_list[i][0][0]
            y = m_2 * x + b_2
            intersection = np.array([x, y])
        elif m_2 is None:
            # Second line is vertical, find intersection
            x = segment_list[k][0][0]
            y = m_1 * x + b_1
            intersection = np.array([x, y])
        else:
            # Neither line is vertical
            if m_1 == m_2:
                # Lines are parallel
                intersection = None
            else:
                # Lines will intersect
                intersection = getIntersection(m_1, b_1, m_2, b_2)

        intersections.append(intersection)
    intersections = np.array(intersections)

    # shift the intersections by the com_diff
    intersections = intersections + com_diff

    # Initialize a set to keep track of plotted translations
    plotted_translations = set()

    # Draw the images first if needed
    if draw_images:
        # Calculate dpm radius from num vertices and vertex radius
        dpm_radius = np.sqrt(max(dpm_df.area.values) / np.pi) * 3

        boundary_conditions = [
            {'condition': True, 'translation': np.array([0, 0])},  # No translation
            {'condition': com_mod[0] < dpm_radius, 'translation': np.array([box_lengths[0], 0])},
            {'condition': com_mod[0] > box_lengths[0] - dpm_radius, 'translation': np.array([-box_lengths[0], 0])},
            {'condition': com_mod[1] < dpm_radius, 'translation': np.array([0, box_lengths[1]])},
            {'condition': com_mod[1] > box_lengths[1] - dpm_radius, 'translation': np.array([0, -box_lengths[1]])},
        ]

        for bc_x in boundary_conditions:
            for bc_y in boundary_conditions:
                if bc_x['condition'] or bc_y['condition']:
                    com_diff = bc_x['translation'] + bc_y['translation']
                    if tuple(com_diff) not in plotted_translations:  # Check if this translation has been plotted
                        polygon = RoundedPolygon(intersections + com_diff, pad=radius / 2, **kwargs)  # TODO vertify correct padding (/2 seems slightly better)
                        ax_anim.add_patch(polygon)
                        plotted_translations.add(tuple(com_diff))  # Add this translation to the set

    # Add the original polygon last to prevent overplotting
    if tuple(np.array([0, 0])) not in plotted_translations:
        polygon = RoundedPolygon(intersections, pad=radius / 2, **kwargs)  # TODO vertify correct padding (/2 seems slightly better)
        ax_anim.add_patch(polygon)

def draw_axes_director_field(ax, dpm_df, box_lengths, draw_minor=False, draw_images=True, **kwargs):

    for dpm_id in dpm_df.dpm_id.unique():
        # get the major and minor axis vectors
        sub_df = getDpmData(dpm_df, dpm_id)
        major_axis_vector = sub_df[['major_axis_x_i', 'major_axis_y_i', 'major_axis_x_f', 'major_axis_y_f']].values[0]
        minor_axis_vector = sub_df[['minor_axis_x_i', 'minor_axis_y_i', 'minor_axis_x_f', 'minor_axis_y_f']].values[0]

        # get the shift vector to put the dpm back in the box
        com_mod = np.mod(sub_df[['x', 'y']].values[0], box_lengths)
        com_shift = sub_df[['x', 'y']].values[0] - com_mod

        # adjust the major and minor axis vectors
        if draw_minor:
            axis = minor_axis_vector
        else:
            axis = major_axis_vector
        axis[:2] -= com_shift[0]
        axis[2:] -= com_shift[1]
        ax.plot(axis[[0, 2]], axis[[1, 3]], **kwargs)

        if draw_images:
            # Calculate dpm radius from num vertices and vertex radius
            dpm_radius = np.sqrt(max(dpm_df.area.values) / np.pi) * 3
            
            # bottom
            if com_mod[1] < dpm_radius:
                ax.plot(axis[[0, 2]], axis[[1, 3]] + box_lengths[1], **kwargs)
            # top
            if com_mod[1] > box_lengths[1] - dpm_radius:
                ax.plot(axis[[0, 2]], axis[[1, 3]] - box_lengths[1], **kwargs)
            # left
            if com_mod[0] < dpm_radius:
                ax.plot(axis[[0, 2]] + box_lengths[0], axis[[1, 3]], **kwargs)
            # right
            if com_mod[0] > box_lengths[0] - dpm_radius:
                ax.plot(axis[[0, 2]] - box_lengths[0], axis[[1, 3]], **kwargs)
            # bottom left
            if com_mod[0] < dpm_radius and com_mod[1] < dpm_radius:
                ax.plot(axis[[0, 2]] + box_lengths[0], axis[[1, 3]] + box_lengths[1], **kwargs)
            # bottom right
            if com_mod[0] > box_lengths[0] - dpm_radius and com_mod[1] < dpm_radius:
                ax.plot(axis[[0, 2]] - box_lengths[0], axis[[1, 3]] + box_lengths[1], **kwargs)
            # top left
            if com_mod[0] < dpm_radius and com_mod[1] > box_lengths[1] - dpm_radius:
                ax.plot(axis[[0, 2]] + box_lengths[0], axis[[1, 3]] - box_lengths[1], **kwargs)
            # top right
            if com_mod[0] > box_lengths[0] - dpm_radius and com_mod[1] > box_lengths[1] - dpm_radius:
                ax.plot(axis[[0, 2]] - box_lengths[0], axis[[1, 3]] - box_lengths[1], **kwargs)


def update_animation(frame, ax_anim, dpm_df, vertex_df, box_lengths, frame_to_step, draw_forces, draw_images, tracer_id=None, draw_director=False):
    """Update the animation.
    
    Args:
        frame (int): frame number
        ax_anim (matplotlib.axes.Axes): Axes object to be configured.
        dpm_df (dataframe): df containing the dpm data
        vertex_df (dataframe): df containing the vertex data
        box_lengths (list): list of box lengths
        steps_to_skip (int): number of steps to skip
        
    Returns:
        None
    """
    ax_anim.clear()
    config_anim_plot(ax_anim, box_lengths, offset=1)
    drawBoxBorders(ax_anim, box_lengths, color='black', linestyle='--', alpha=0.5)
    # get a color map for the dpm ids
    dpm_ids = dpm_df.dpm_id.unique()
    if len(dpm_ids) > 1:
        dpm_id_to_color = getValToColorMap(dpm_ids)
    else:
        dpm_id_to_color = getValToColorMap([0, 1])
    step = frame_to_step[frame]
    for dpm_id in dpm_ids:
        color = dpm_id_to_color[dpm_id]
        if tracer_id is not None and dpm_id == tracer_id:
            drawDpm(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, alpha=0.5, color=color)
            if draw_forces:
                drawDpmForces(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, alpha=0.5, color='blue')
        else:
            drawDpmAsPolygon(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, draw_images, alpha=0.5, facecolor=color, edgecolor='black')
            if draw_forces:
                drawDpmForces(ax_anim, vertex_df, dpm_df, dpm_id, step, box_lengths, alpha=0.5)
    if draw_director:
        draw_axes_director_field(ax_anim, getAtStep(dpm_df, step), box_lengths, draw_minor=False, draw_images=True, color='k', alpha=0.5)
            

def animate_dpm_data(dpm_df, vertex_df, config_df, num_frames, path, draw_images, draw_forces=False, draw_director=False, tracer_id=None, **kwargs):
    """Animate the dpm data.
    
    Args:
        dpm_df (dataframe): df containing the dpm data
        vertex_df (dataframe): df containing the vertex data
        box_lengths (list): list of box lengths
        num_frames (int): number of frames
        
    Returns:
        animation: animation object
    """

    if num_frames >= dpm_df.step.nunique():
        num_frames = dpm_df.step.nunique()

    steps = dpm_df.step.unique()
    frame_to_step = steps[::len(steps) // num_frames]

    box_lengths = config_df[['Lx', 'Ly']].values[0]
    fig, axes = initialize_plot(1, box_lengths, offset=1)
    
    with tqdm(total=num_frames, desc='Animating') as pbar:
        anim = animation.FuncAnimation(
            fig,
            update_animation,
            fargs=(axes[0], dpm_df, vertex_df, box_lengths, frame_to_step, draw_forces, draw_images, tracer_id, draw_director),
            frames=num_frames,
            interval=100,
            blit=False
        )
        anim.save(path, progress_callback=lambda i, n: pbar.update(1), **kwargs)
        pbar.close()   