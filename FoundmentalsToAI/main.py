class State:
    def __init__(self, monk, savage, boat, f: int):
        self.monk = monk
        self.savage = savage
        self.boat = boat
        self.f = f

    def __lt__(self, other):
        return self.f < other.f


s1 = State(0, 0, 0, 3)
s2 = State(0, 0, 0, 2)
s = s1 < s2
import queue

q = queue.PriorityQueue()
q.put(State(0, 0, 0, 3))
q.put(State(0, 0, 0, 2))
q.put(State(0, 0, 0, -1))
q.put(State(0, 0, 0, 10))
q.put(State(0, 0, 0, 14))
sss = [State(0, 0, 0, 3), 1]
s1 = State(0, 0, 0, 3)
print(1 in sss)