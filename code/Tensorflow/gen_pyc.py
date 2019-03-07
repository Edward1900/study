#test_pyc

import readxml as rd
import py_compile
py_compile.compile('pytest.py')


data=rd.readxml('dat.xml')
print(data)