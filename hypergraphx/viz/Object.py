from typing import List, Tuple
import numpy as np

class Object:
    def __init__(self, Cartesian_coords_list: List[Tuple[float, float]]):
        self.Cartesian_coords_list = np.array(Cartesian_coords_list)

    def Smooth_by_Chaikin(self, number_of_refinements) -> List[Tuple[float, float]]:
        # check if the list of coordinates provided represents a polygon
        is_polygon = self.Cartesian_coords_list[0][0] == self.Cartesian_coords_list[-1][0]\
                        and self.Cartesian_coords_list[0][1] == self.Cartesian_coords_list[-1][1]
        # repeat the refinement process k times
        for i in range(number_of_refinements):
            self.New_cartesian_coords_list = list()
            last_pt = self.Cartesian_coords_list[-1]
            # for each point in the coords list, and 
            for pt1 in self.Cartesian_coords_list[:-1]:
                # add 2 points to the new coord list:
                #    one 1/4th of the way between pt1 and the last point
                Q = 0.75 * pt1 + 0.25 * last_pt
                self.New_cartesian_coords_list.append(Q)

                #    one 3/4ths of the way between pt1 and the last point
                R = 0.25 * pt1 + 0.75 * last_pt
                self.New_cartesian_coords_list.append(R)

            # if the original list of points represented a polygon
            if is_polygon:
                # add the first point to the end of the new list
                self.New_cartesian_coords_list.append(self.New_cartesian_coords_list[0])

            self.Cartesian_coords_list = self.New_cartesian_coords_list

        # cast the points back to tuples and return their list
        return [(float(pt[0]), float(pt[1])) for pt in self.Cartesian_coords_list]