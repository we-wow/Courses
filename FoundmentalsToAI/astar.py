"""

@Author : Wei Mingjiang
@Time   : 2022/3/24 20:31
@File   : astar.py
@Version: 0.1.0
@Content: First version.
"""
import queue
import time


class State:
    def __init__(self, monk, savage, boat, f: int = 0):
        self.monk = monk
        self.savage = savage
        self.boat = boat
        self.f = f
        self.father = None
        self.children = []
        self.index = f"{self.monk}-{self.savage}-{self.boat}"

    def __lt__(self, other):
        return self.f < other.f


class MonkAndSavage:
    def __init__(self, total_monk=3, total_savage=3, boat_capacity=2):
        self.total_monk = total_monk
        self.total_savage = total_savage
        self.boat_capacity = boat_capacity

    def next_state(self, s: State, mob, sob):
        if s.boat and (s.monk < mob or s.savage < sob):
            return None
        elif not s.boat and (self.total_monk - s.monk < mob or self.total_savage - s.savage < sob):
            return None
        _f = (-1) ** s.boat  # direction factor
        left_mob = s.monk + _f * mob
        left_savage = s.savage + _f * sob
        right_mob = self.total_monk - s.monk - _f * mob
        right_savage = self.total_savage - s.savage - _f * sob
        if (left_mob and left_mob < left_savage) or (right_mob and right_mob < right_savage):
            return None
        return State(left_mob, left_savage, 1 ^ s.boat)

    def generate_action_set(self):
        action_sets = []
        for m in range(self.boat_capacity + 1):
            for s in range(self.boat_capacity + 1):
                if (m + s > self.boat_capacity) or (m + s < 1) or (m and m < s):
                    continue
                action_sets.append((m, s))
        return action_sets

    def a_star_search(self):
        t0 = time.time()
        que = queue.PriorityQueue()
        que.put(State(self.total_monk, self.total_savage, 1, 0))
        visited = {}
        static_action_set = self.generate_action_set()
        solution_found = False
        while not que.empty():
            s = que.get()
            if not visited.__contains__(s.index):
                visited[s.index] = s
            else:
                continue
            if s.monk + s.savage + s.boat == 0:
                solution_found = True
                break
            for monk_on_boat, savage_on_boat in static_action_set:
                s_ = self.next_state(s, monk_on_boat, savage_on_boat)
                if s_ is None:
                    continue
                s_.father = s.index
                s.children.append(s_.index)
                s_.f = len(visited) + s_.monk + s_.savage - 2 * s.boat
                que.put(s_)
        t1 = time.time()
        if not solution_found:
            return [], t1 - t0
        return visited, t1 - t0


if __name__ == '__main__':
    game = MonkAndSavage(total_monk=20, total_savage=20, boat_capacity=4)
    res, time_used = game.a_star_search()
    if not res:
        print("Solution Not Found")
    else:
        print(f"Solution Found: {time_used:.4f} second(s)")
        state = res["0-0-0"]
        solution = [state, ]
        while state.father is not None:
            state = res[state.father]
            solution.append(state)
        solution.reverse()
        for step, i in zip(solution, range(len(solution))):
            if i < len(solution) - 1:
                if step.boat:
                    msg = f"{step.monk - solution[i + 1].monk} monk(s), " \
                          f"{step.savage - solution[i + 1].savage} savage(s) "
                else:
                    msg = f"{solution[i + 1].monk - step.monk} monk(s), " \
                          f"{ solution[i + 1].savage - step.savage} savage(s)"
            else:
                msg = "Done"
            print(f"[{i+1:02d}] {step.monk, step.savage, step.boat} {msg}")


