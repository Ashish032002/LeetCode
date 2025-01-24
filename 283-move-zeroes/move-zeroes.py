class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        a  = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[a] = nums[i]
                a +=1 
        
        while(a<len(nums)):
            nums[a] = 0
            a += 1    
        