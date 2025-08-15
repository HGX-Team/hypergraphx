from typing import List, Tuple
import numpy as np

class Object:
    def __init__(self, Cartesian_coords_list: List[Tuple[float, float]]):
        self.Cartesian_coords_list = np.array(Cartesian_coords_list)

    def Smooth_by_Chaikin(self, number_of_refinements: int) -> List[Tuple[float, float]]:
        coords = np.array(self.Cartesian_coords_list, dtype=float)

        # Check if closed polygon (first point == last point)
        if not np.allclose(coords[0], coords[-1]):
            raise ValueError("coordinate list passed to Smooth_by_Chaikin represents an open curve, expected a polygon")

        for _ in range(number_of_refinements):
            new_coords = []

            # Wrap around so last point connects to first
            pairs = zip(coords, np.roll(coords, -1, axis=0))

            for p1, p2 in pairs:
                p1 = np.asarray(p1)
                p2 = np.asarray(p2)

                # Q: 1/4 of the way toward p2
                Q = 0.75 * p1 + 0.25 * p2
                # R: 3/4 of the way toward p2
                R = 0.25 * p1 + 0.75 * p2

                new_coords.append(Q)
                new_coords.append(R)

            coords = np.array(new_coords)

        return [tuple(map(float, pt)) for pt in coords]