import numpy
import math

print ('yolo!')
b = \
'''
this is fucking studpid
u know?
'''
a = 'yolo!';
print (b[0:] + b[-10:])

#input("\n\nPress the enter key to exit.")

a = b = c = 1
print(a, b, c)
a = 2
print(a, b, c)

print (r'hiiiir\niiiii')


def printme( str ):
   "This prints a passed string into this function"
   print str
   return

printme('yo')

# Function definition is here
def printinfo( arg1, *vartuple ):
   "This prints a variable passed arguments"
   print "Output is: "
   arg1 = math.ceil(arg1);
   print math.ceil(arg1)
   #for var in vartuple:
   #   print var;
   return;

# Now you can call printinfo function
x = 10.123;
printinfo( x );
print (x);
printinfo( 70, 60, 50 )