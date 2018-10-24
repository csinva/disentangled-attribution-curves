def intersect(i1, i2):
    if(i1[0] <= i2[0] and i1[1] >= i2[1]):
        ints = [(i1[0], i2[0], i1[2]), (i2[0], i2[1], i1[2] + i2[2]), (i2[1], i1[1], i1[2])]
        valid = []
        for i in ints:
            if(i[1] != i[0]):
                valid.append(i)
        return valid
    elif(i2[0] <= i1[0] and i2[1] >= i1[1]):
        ints = [(i2[0], i1[0], i2[2]), (i1[0], i1[1], i2[2] + i1[2]), (i1[1], i2[1], i2[2])]
        valid = []
        for i in ints:
            if(i[1] - i[0] != 0):
                valid.append(i)
        return valid
    elif(i1[0] < i2[0]):
        return [(i1[0], i2[0], i1[2]), (i2[0], i1[1], i1[2] + i2[2]), (i1[1], i2[1], i2[2])]
    else:
        return [(i2[0], i1[0], i2[2]), (i1[0], i2[1], i2[2] + i1[2]), (i2[1], i1[1], i2[2])]

def merge_two_1d(f1, f2):
    if(len(f1) == 0):
        return f2
    elif(len(f2) == 0):
        return f1
    merged = intersect(f1[0], f2[0])
    i, j = 1, 1
    while(i < len(f1) and j < len(f2)):
        last = merged[-1]
        tail = last
        if(f1[i] < f2[j]):
            tail = intersect(last, f1[i])
            i += 1
        else:
            tail = intersect(last, f2[j])
            j += 1
        merged = merged[:-1] + tail
    while(i < len(f1)):
        last = merged[-1]
        tail = intersect(last, f1[i])
        merged = merged[:-1] + tail
        i += 1
    while(j < len(f2)):
        last = merged[-1]
        tail = intersect(last, f2[j])
        merged = merged[:-1] + tail
        j += 1
    return merged

"""
PARAMETERS:
functions: a list of lists, each sublist representing the values of the piecewise constant functions to be combined.  Each
element in the sublists is of the form (min, max, vals), where min and max represent the bounds of an
interval, and val represents the function's value over that interval (as an array of size 1)
OUTPUT:
a list of intervals and values that define a piecewise constant function, similar to the input.
"""
def piecewise_average_1d(functions):
    if(len(functions) <= 1):
        return functions
    functions[0].sort()
    functions[1].sort()
    merged = merge_two_1d(functions[0], functions[1])
    for i in range(2, len(functions)):
        functions[i].sort()
        merged = merge_two_1d(functions[i], merged)
    averaged = []
    for i in merged:
        avg = (i[0], i[1], sum(i[2])/len(i[2]))
        averaged.append(avg)
    return averaged

def test_avg():
    piecewise_1 = [(- float('inf'), -1, [0]), (-1, 3, [2]), (3, float('inf'), [1])]
    print("1", piecewise_1)
    piecewise_2 = [(- float('inf'), -2, [1]), (-2, 0, [0]), (0, 4, [2]), (4, float('inf'), [0])]
    print("2", piecewise_2)
    avg = piecewise_average_1d([piecewise_1, piecewise_2])
    print("avg", avg)
test_avg()
