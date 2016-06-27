#!/usr/bin/python

# Open a file
fo = open("foo.txt", "wb")
fo.write( "Python is a great language.\nYeah its great!!\n");

# Close opend file
fo.close()

import os

# Changing a directory to "/home/newdir"
os.chdir("./Phone")
