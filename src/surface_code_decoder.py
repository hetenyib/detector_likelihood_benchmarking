# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import networkx as nx
import stim
import pymatching
import math
from typing import Callable, List, Iterable, Dict

### The following three functions below have been copied from the built-in method of stim: stim.Circuit.generated("surface_code:rotated_memory_z",...)
def iter_flatten_model(model: stim.DetectorErrorModel,
                    handle_error: Callable[[float, List[int], List[int]], None],
                    handle_detector_coords: Callable[[int, np.ndarray], None]):
    det_offset = 0
    coords_offset = np.zeros(100, dtype=np.float64)

    def _helper(m: stim.DetectorErrorModel, reps: int):
        nonlocal det_offset
        nonlocal coords_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _helper(instruction.body_copy(), instruction.repeat_count)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets: List[int] = []
                        frames: List[int] = []
                        t: stim.DemTarget
                        p = instruction.args_copy()[0]
                        for t in instruction.targets_copy():
                            if t.is_relative_detector_id():
                                dets.append(t.val + det_offset)
                            elif t.is_logical_observable_id():
                                frames.append(t.val)
                            elif t.is_separator():
                                # Treat each component of a decomposed error as an independent error.
                                # (Ideally we could configure some sort of correlated analysis; oh well.)
                                handle_error(p, dets, frames)
                                frames = []
                                dets = []
                        # Handle last component.
                        handle_error(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                        a = np.array(instruction.args_copy())
                        coords_offset[:len(a)] += a
                    elif instruction.type == "detector":
                        a = np.array(instruction.args_copy())
                        for t in instruction.targets_copy():
                            handle_detector_coords(t.val + det_offset, a + coords_offset[:len(a)])
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
    _helper(model, 1)


def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel) -> 'nx.Graph':
    """Convert a stim error model into a NetworkX graph."""

    # Local import to reduce sinter's startup time.
    try:
        import networkx as nx
    except ImportError as ex:
        raise ImportError(
            "pymatching was installed without networkx?"
            "Run `pip install networkx`.\n"
        ) from ex

    g = nx.Graph()
    boundary_node = model.num_detectors
    g.add_node(boundary_node, is_boundary=True, coords=[-1, -1, -1])

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets = [dets[0], boundary_node]
        if len(dets) > 2:
            print("Warning: hyperedge with detectors "+str(dets)+" is ignored, logical change: "+str(bool(frame_changes)))
            return
            # raise NotImplementedError(
            #     f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = g.get_edge_data(*dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["fault_ids"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if set(old_frame_changes) == set(frame_changes):
                p = p * (1 - old_p) + old_p * (1 - p)
                g.remove_edge(*dets)
        if p > 0.5:
            p = 1 - p
        if p > 0:
            g.add_edge(*dets, weight=math.log((1 - p) / p), fault_ids=frame_changes, error_probability=p)

    def handle_detector_coords(detector: int, coords: np.ndarray):
        g.add_node(detector, coords=coords)

    iter_flatten_model(model, handle_error=handle_error, handle_detector_coords=handle_detector_coords)

    return g


def detector_error_model_to_pymatching_graph(model: stim.DetectorErrorModel) -> 'pymatching.Matching':
    """Convert a stim error model into a pymatching graph."""

    # Local import to reduce sinter's startup time.
    import pymatching

    g = detector_error_model_to_nx_graph(model)
    num_detectors = model.num_detectors
    num_observables = model.num_observables

    # Ensure invincible detectors are seen by explicitly adding a node for each detector.
    for k in range(num_detectors):
        g.add_node(k)
    # Ensure invincible observables are seen by adding a boundary edge with all observables.
    g.add_node(num_detectors + 1)
    g.add_edge(num_detectors, num_detectors + 1, weight=1, fault_ids=list(range(num_observables)))

    return pymatching.Matching(g)