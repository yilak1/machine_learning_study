'''
effective python1
'''
import os
from urllib.parse import parse_qs
# python3 bytes and str

def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value  #Instance of str

def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value  #Instance of bytes

'''with open('test.txt','w') as f:
    f.write(os.urandom(10)) #TypeError: write() argument must be str, not bytes'''

with open('test.txt','wb') as f:
    f.write(os.urandom(10)) #Using bytes to open

my_values = parse_qs('red=5&blue=0&green=', keep_blank_values=True)
print(repr(my_values))      #repr一般将对象转换成字符串，str则是将数值转换成字符串
print('Red    ', my_values.get('red'))
print('Blue    ', my_values.get('blue'))
print('Opacity    ', my_values.get('opacity'))    # get获得字典的键值
