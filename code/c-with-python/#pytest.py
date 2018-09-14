#test function
import tensorflow as tf
import numpy as np
def add(a,b):
    print("in python function add")
    print("a = " + str(a))
    print("b = " + str(b))
    print("ret = " + str(a+b))
    return str(a+b)
def foo(a):
    print("in python function foo")
    print("a = " + str(a))
    print("ret = " + str(a * a))
    return a * a

sess = tf.Session()
