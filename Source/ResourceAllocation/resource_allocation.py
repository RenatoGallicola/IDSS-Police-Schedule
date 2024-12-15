from ortools.linear_solver import pywraplp
from location_generator import Location

def allocate_policemen(loc_dist, loc_req, mins_threshold, margin, vel):

    # Average velocity of policemen in km/h
    avg_vel = vel

    dist_threshold = mins_threshold * avg_vel / 60.0
    
    num_policemen = len(loc_dist)
    num_locations = len(loc_req)

    max_dist = int(((Location.MAXLAT.value - Location.MINLAT.value)**2 + (Location.MAXLONG.value - Location.MINLONG.value)**2)**0.5)
    delay_coeff = max_dist * num_policemen

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None

    # x[i][j] = 1 if policeman i is assigned to location j, else 0
    x = []
    for i in range(num_policemen):
        x.append([solver.NumVar(0, 1, f'x[{i},{j}]') for j in range(num_locations)])

    # Minimize total distance and number of late policemen
    objective = solver.Objective()
    for i in range(num_policemen):
        for j in range(num_locations):
            objective.SetCoefficient(x[i][j], loc_dist[i][j])
    objective.SetMinimization()

    # Each policeman is assigned to exactly one location
    for i in range(num_policemen):
        solver.Add(solver.Sum([x[i][j] for j in range(num_locations)]) == 1)

    # Each location receives the required number of policemen
    for j in range(num_locations):
        if (loc_req[j] > 0):
            solver.Add(solver.Sum([x[i][j] for i in range(num_policemen)]) >= loc_req[j])
            solver.Add(solver.Sum([x[i][j] for i in range(num_policemen)]) <= loc_req[j] + margin)
        else:
            solver.Add(solver.Sum([x[i][j] for i in range(num_policemen)]) == 0)
          
    # Compute the number of late policemen
    late_policemen = []
    for i in range(num_policemen):
        late_policemen.append(solver.NumVar(0, 1, f'late_policeman[{i}]'))

    for i in range(num_policemen):
        solver.Add(late_policemen[i] >= solver.Sum([x[i][j] * (loc_dist[i][j] > dist_threshold) for j in range(num_locations)]))

    # Combine late policemen and distance 
    #solver.Minimize(delay_coeff*solver.Sum(late_policemen) + solver.Sum([x[i][j] * loc_dist[i][j] for i in range(num_policemen) for j in range(num_locations)]))
    #solver.Minimize(solver.Sum([x[i][j] * loc_dist[i][j] for i in range(num_policemen) for j in range(num_locations)]))
    solver.Minimize(delay_coeff*solver.Sum(late_policemen) + solver.Sum([x[i][j] * loc_dist[i][j] for i in range(num_policemen) for j in range(num_locations)]))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # tot = solver.Objective().Value()
        allocation = [[int(x[i][j].solution_value()) for j in range(num_locations)] for i in range(num_policemen)]
        final_dest = []
        for i in range(num_policemen):
            for j in range(num_locations):
                if allocation[i][j] == 1:
                    final_dest.append(j)
                    break
        police_dist = [loc_dist[i][final_dest[i]] for i in range(num_policemen)]
        return final_dest, police_dist
    else:
        return None