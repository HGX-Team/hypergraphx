from typing import List, Tuple
import numpy as np

class Object:
    def __init__(self, Cartesian_coords_list: List[Tuple[int, int]]):
        self.Cartesian_coords_list = Cartesian_coords_list

    def Smooth_by_Chaikin(self, number_of_refinements) -> List:
        # check if the list of coordinates provided represents a polygon
        is_polygon = self.Cartesian_coords_list[0] == self.Cartesian_coords_list[-1]
        # repeat the refinement process k times
        for i in range(number_of_refinements):
            self.New_cartesian_coords_list = list()
            # for each point in the coords list
            for pt1 in self.Cartesian_coords_list[:-1]:
                pt1 = np.array(pt1)
                # also get the last point
                pt2 = np.array(self.Cartesian_coords_list[-1])
                # add 2 points:
                #    one 1/4th of the way between pt1 and the last point
                Q = tuple(0.75 * pt1 + 0.25 * pt2)
                self.New_cartesian_coords_list.append(Q)
                #    one 3/4ths of the way between pt1 and the last point
                R = tuple(0.25 * pt1 + 0.75 * pt2)
                self.New_cartesian_coords_list.append(R)

            # if the original list of points represented a polygon
            if is_polygon:
                # add the first point to the end of the new list
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list

        return self.Cartesian_coords_list