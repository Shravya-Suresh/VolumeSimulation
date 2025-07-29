def adding (a, b):
    res = a + b
    return res

def subtract (a, b):
    res = a - b
    return res

a = int(input("enter a number to add"))
b = int(input("enter another number to add"))
res1 = adding(a, b)
res2 = subtract(a, b)
print("sum: " + res1)
print("difference: "+ res2)