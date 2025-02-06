class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left , max_length , zerocount = 0, 0 , 0

        for right in range(len(nums)):
            if nums[right] == 0:
                zerocount += 1
            while zerocount > k:
                if nums[left] == 0:
                    zerocount -= 1
                left += 1
            
            max_length = max(max_length , (right - left + 1))

        return max_length