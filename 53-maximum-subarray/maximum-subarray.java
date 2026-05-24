class Solution {
    public int maxSubArray(int[] nums) {

        int current_sum = nums[0];
        int total_sum = current_sum;

        for(int i = 1 ; i < nums.length ; i ++){

            total_sum = Math.max(total_sum + nums[i], nums[i]);
            current_sum = Math.max(total_sum , current_sum);
              
        }    
    return current_sum;
    }
}