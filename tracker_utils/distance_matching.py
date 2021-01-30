# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (
            x[1] - y[1]) ** 2)


def dist(bbox, candidates):
    distances = [distance(bbox, candidate) for candidate in candidates]
    return np.array(distances)


def dist_cost(tracks, detections, track_indices=None,
              detection_indices=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        # if tracks[track_idx].time_since_update > 1:
        #     cost_matrix[row, :] = linear_assignment.INFTY_COST
        #     continue

        bbox = tracks[track_idx].get_center()
        candidates = np.asarray([detections[i].get_center() for i in detection_indices])
        cost_matrix[row, :] = dist(bbox, candidates)
    return cost_matrix
