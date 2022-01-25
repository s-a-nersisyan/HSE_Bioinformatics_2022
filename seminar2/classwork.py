# Task 1
x = float(input())
n = int(input())
for i in range(n):
    if x == float(input()):
        print("Found!")
        break

# Task 2
x = [1, 3, 5, 6]
y = [2, 4, 7, 8]

result = []
i, j = 0, 0
while i < len(x) or j < len(y):
    if i < len(x) and (j == len(y) or x[i] < y[j]):
        result.append(x[i])
        i += 1
    else:
        result.append(y[j])
        j += 1

print(result)

# Task 3: brute force
arr = [7, 7, 7, 7, 7, 5, 5, 5, 6, 6, 7]
for x in arr:
    count = 0
    for y in arr:
        if x == y:
            count += 1
    
    if count >= len(arr) / 2:
        print(x)
        break

# Task 3: nice algorithm
a = 0
while 1:
    x = input()
    if x == "Stop":
        break
    
    x = int(x)

    if a == 0:
        candidate = x
    
    if x == candidate:
        a += 1
    else:
        a -= 1

    print(x, candidate, a)

print("Result: ", candidate)
