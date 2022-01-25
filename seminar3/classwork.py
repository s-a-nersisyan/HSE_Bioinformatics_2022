import time

# Task 1: merge sort
def merge(x, y):
    result = []
    i, j = 0, 0
    while i < len(x) or j < len(y):
        if i < len(x) and (j == len(y) or x[i] < y[j]):
            result.append(x[i])
            i += 1
        else:
            result.append(y[j])
            j += 1
    
    return result


def merge_sort(arr):
    if len(arr) in [0, 1]:
        return arr

    return merge(merge_sort(arr[:len(arr)//2]), merge_sort(arr[len(arr)//2:]))


# Task 2: counting sort of digits
def sort_digits(arr):
    counts = [0]*10
    for x in arr:
        counts[x] += 1
    
    result = []
    for i in range(10):
        result += [i]*counts[i]

    return result

start = time.time()
merge_sort(list(range(10)) * 100000)
end = time.time()
print(end - start)

start = time.time()
sorted(list(range(10)) * 100000)
end = time.time()
print(end - start)


# Task 3: dynamic programming
c = [1, 2, 3, 40, 5, 6]
opt = [c[0], c[1]]
for i in range(2, len(c)):
    opt.append(min(opt[i - 1], opt[i - 2]) + c[i])

path = []
i = len(c) - 1
while i >= 0:
    path.append(i)
    if opt[i - 1] < opt[i - 2]:
        i -= 1
    else:
        i -= 2

print(c)
print(opt)
print(path[::-1])
