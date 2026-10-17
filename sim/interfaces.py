"""Inputs and outputs of the Mapper and the allocator. Plain dataclasses, no ROS2.

Data flow (one box per future ROS2 node; the sim drives the same classes directly):

    sensor/vision --FrameReport--> Mapper (owns ALL state) --AllocatorInput--> allocator (stateless)
                                     ^                                            |
                                     +------------- Allocation <-----------------+
    drones --DroneStatus--> Mapper                          Allocation --> drone path planner

- The Mapper is the only stateful component: belief maps, per-drone drift estimates, current
  allocations, and each drone's remaining trajectory. It is fed FrameReports and DroneStatus updates,
  records Allocations, and produces an AllocatorInput snapshot on demand.
- The allocator is a pure function: allocate(AllocatorInput) -> Allocation. Same input, same output
  (given a fixed seed for stochastic policies). It never keeps state between calls.
- Units: feet and seconds; world frame and grids as in sim/geometry.py.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    xy: tuple[float, float]  # ground position estimate in the reporting drone's believed (GPS) frame, ft
    confidence: float = 1.0  # detector confidence 0..1


@dataclass
class FrameReport:
    """One camera frame from one drone: what it looked at and what it saw there."""
    drone_id: int
    t: float                         # seconds since run start
    pose_xy: tuple[float, float]     # believed (GPS) position of the drone, ft
    yaw: float                       # heading, rad (0 = +x, pi/2 = +y)
    altitude_ft: float
    pos_sigma_ft: float              # 1-sigma position uncertainty the drone reports for this frame
    footprint_ft: tuple[float, float]  # (across, along) ground footprint of the camera at this altitude
    detections: list[Detection] = field(default_factory=list)  # empty = looked and saw nothing


@dataclass
class DroneStatus:
    """Latest state of one drone, as reported by the drone/controller."""
    drone_id: int
    t: float
    pose_xy: tuple[float, float]          # believed position, ft
    yaw: float
    remaining_path: np.ndarray            # (M, 2) future path points in ft, nearest first; empty when idle
    remaining_eta_s: np.ndarray           # (M,) seconds until each remaining_path point is reached
    idle: bool                            # True = finished its allocation and needs a new one


@dataclass
class AllocatorInput:
    """Snapshot handed to the allocator when drone `drone_id` needs work. All grids [row=y, col=x]."""
    drone_id: int
    t: float                    # seconds since run start
    t_end: float                # run deadline (seconds since start), e.g. 420
    p_mine: np.ndarray          # (300, 80) float32: per 1 ft cell, P(the square's mine is in this cell)
    seen: np.ndarray            # (300, 80) float32: 0..1 confidence the cell was observed
    allocated: np.ndarray       # (300, 80) int8: drone id holding the cell, -1 = none
    others_path_eta: np.ndarray  # (300, 80) float32: 1 = another drone's path reaches the cell now, fading to 0 at route end; 0 = none
    drones: list[DroneStatus]   # all drones, indexed by drone_id
    pos_sigma_ft: float         # current position-uncertainty level (the drift curriculum's sigma in the sim)
    block_mask: np.ndarray      # (N_BLOCKS,) bool: True = this block may be assigned to drone_id


@dataclass
class Allocation:
    """The allocator's answer: blocks for one drone, in the order the policy picked them.
    The drone's own planner decides visit order (exact TSP / ACO) and the actual path."""
    drone_id: int
    blocks: list[int]            # block ids (sim/geometry.py), len = K

    def squares(self) -> list[tuple[int, int]]:
        """The allocation expanded to scoring squares (row, col), for consumers that want cells."""
        from sim.geometry import BLK_FT, SQ_FT, block_bounds
        per = int(BLK_FT // SQ_FT)
        out = []
        for b in self.blocks:
            x0, y0, _, _ = block_bounds(b)
            r0, c0 = int(y0 // SQ_FT), int(x0 // SQ_FT)
            out += [(r0 + i, c0 + j) for i in range(per) for j in range(per)]
        return out
