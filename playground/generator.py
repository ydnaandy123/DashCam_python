import sys
def fibonacci(n): #generator function
    a, b, counter = 0, 1, 0
    print a
    while True:
        if (counter > n): 
            return
        #yield a
        a, b = b, a + b
        counter += 1
    print a
    return a
    
f = fibonacci(6) #f is iterator object
print f
'''
while True:
   try:
      print (next(f))
   except StopIteration:
      sys.exit()
      '''
