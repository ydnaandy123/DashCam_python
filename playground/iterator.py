list=[1,2,3,4]
it = iter(list) # this builds an iterator object
print (next(it)) #prints next available element in iterator
#Iterator object can be traversed using regular for statement
#!usr//bin/python3
for x in it:
   print x, ' 3'
#or using next() function
it = iter(list)
while True:
   try:
      print (next(it))
   except StopIteration:
      print ('end')
      break
