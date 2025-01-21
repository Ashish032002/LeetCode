class Solution(object):
    def gcd(self, x, y):
        if y == 0:
            return x  
        return self.gcd(y, x % y)
        
    def gcdOfStrings(self, str1, str2):
        if str1 + str2 != str2 + str1:
            return ""
        
        max_length = self.gcd(len(str1), len(str2))  
        return str1[:max_length]
