"""
define an INTERVAL as a tuple, where the entry at index 0 is the lower bound, and the entry at index 1
is the upper bound, both exclusive.
"""
def interval_intersect(a, b):
    if(min(a) == -float('inf') and min(b) == -float('inf')):
        return True
    elif(max(a) == float('inf') and max(b) == float('inf')):
        return True
    if(min(a) == -float('inf')):
        return min(b) < max(a)
    elif(min(b) == -float('inf')):
        return min(a) < max(b)
    elif(max(a) == float('inf')):
        return min(a) < max(b)
    elif(max(b) == float('inf')):
        return min(b) < max(a)
    radius_a = (a[1] - a[0])/2
    radius_b = (b[1] - b[0])/2
    centroid_a = radius_a + a[0]
    centroid_b = radius_b + b[0]
    return np.abs(centroid_b - centroid_a) < radius_a + radius_b

def join_intervals(a, b):
    if(not interval_intersect(a, b)):
        return -1
    return (max(a[0], b[0]), min(a[1], b[1]))

def point_in_intervals(p, intervals):
    for i in intervals:
        if p > i[0] and p < i[1]:
            return 1
    return 0

def test_intervals():
    i1 = (0, 3)
    i2 = (2, 4)
    i3 = (3, 10)
    i4 = (-1, 2)
    print(i1, "intersects with", i2, "?", interval_intersect(i1, i2))
    print(i1, "intersects with", i3, "?", interval_intersect(i1, i3))
    print(i3, "intersects with", i4, "?", interval_intersect(i3, i4))
    print(i3, "intersects with", i2, "?", interval_intersect(i3, i2))
    print(i4, "intersects with", i1, "?", interval_intersect(i4, i1))
    print("joining", i1, "and", i2, ":", join_intervals(i1, i2))
    print("joining", i3, "and", i2, ":", join_intervals(i3, i2))
    print("joining", i1, "and", i4, ":", join_intervals(i1, i4))
