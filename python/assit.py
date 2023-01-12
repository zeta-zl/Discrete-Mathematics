import math

import unvcode

s1 = "这个网址可以把你的字替换成相近的字，看起来差不多但是查重就查不到"
s2 = "这个⽹址可以把你的字替换成相近的字，看起来差不多但是查重就查不到"
print(unvcode.unvcode(s1, mse=10000))
