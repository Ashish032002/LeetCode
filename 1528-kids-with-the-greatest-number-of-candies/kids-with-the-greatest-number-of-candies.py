class Solution(object):
    def kidsWithCandies(self, candies, extraCandies):
        result = []
        max_no = max(candies)
        for candy in candies:
            if (candy + extraCandies) >= max_no:
                result.append(True)
            else:
                result.append(False)
        
        return result
             
        
        
        


        
        