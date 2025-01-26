class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        current = sum(nums[:k])
        maxx = current
        for i in range(1, len(nums)- k + 1):
            current = current - nums[i-1] + nums[i + k -1]
            maxx = max(maxx,current)
        
        return maxx / k


            